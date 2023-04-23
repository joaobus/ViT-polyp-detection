import torch 
import wandb
import numpy as np

from tqdm import tqdm
from model.modeling import VisionTransformer
from model.data_augmentation import DataAugmentation 
from utils.configs import get_model_config, get_training_config, get_loading_config
from torch.optim import Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall, binary_auroc


class ViTClassifier:
    '''
    General purpose class for building, fitting, predicting and evaluating
    '''
    def __init__(self, get_test_config: bool = False):
        self.model_configs = get_model_config(get_test_config)
        self.training_configs = get_training_config(get_test_config)
        self.loading_configs = get_loading_config()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
        
        self.device = device

        self.model = VisionTransformer(in_channels = self.model_configs['c'],
                                        patch_size = self.model_configs['patch_size'],
                                        emb_size = self.model_configs['emb_size'],
                                        img_size = self.model_configs['h'],
                                        drop_p = self.model_configs['drop_rate'],
                                        forward_drop_p= self.model_configs['drop_rate'],
                                        num_heads = self.model_configs['num_heads'],
                                        n_blocks = self.model_configs['num_blocks'],
                                        n_classes = 1,
                                        device = self.device,
                                        return_attn_weights=self.model_configs['return_attn_weights']
                                        ).to(self.device)

        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), 
                              lr=self.training_configs['lr'], 
                              weight_decay=self.training_configs['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=self.training_configs['lr_patience'])
        self.augment = DataAugmentation()
        

    def load_pretrained(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device)['model_state_dict'])

    
    def get_metrics(self, output: torch.Tensor, label: torch.Tensor):                
        accuracy = binary_accuracy(output, label, threshold = self.training_configs['threshold']).float()
        precision = binary_precision(output, label, threshold = self.training_configs['threshold']).float()
        recall = binary_recall(output, label, threshold = self.training_configs['threshold']).float() 
        f1 = binary_f1_score(output, label, threshold = self.training_configs['threshold']).float() 
        auroc = binary_auroc(output, label).float()
        
        return np.array([accuracy, precision, recall, f1, auroc])
    
    
        
    def fit(self, train_dataloader, val_dataloader, log: bool = True):
        
        if log:
            wandb.login()
    
            wandb.init(
                project="ViTClassifier-Colorretal",
    
                config={
                "learning_rate": self.training_configs['lr'],
                "batch_size": self.loading_configs['batch_size'],
                "architecture": "ViT",
                "dataset": "CP-CHILD",
                "epochs": self.training_configs['num_epochs'],
                "dropout_rate": self.model_configs['drop_rate'],
                "num_heads": self.model_configs['num_heads'],
                "num_blocks": self.model_configs['num_blocks'],
                "weight_decay": self.training_configs['weight_decay'],
                "pos_threshold": self.training_configs['threshold'],
                "lr_patience": self.training_configs['lr_patience']
                }
            )

        best_val_loss = np.inf

        for epoch in range(self.training_configs['num_epochs']):
            accuracy = loss = precision = 0.
            recall = f1 = auroc = 0.
            
            train_metrics = np.array([accuracy, precision, recall, f1, auroc])

            for data, label in tqdm(train_dataloader): 
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.model(self.augment(data))[:,0]
                loss = self.criterion(output, label.float())
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss += loss.float() / len(train_dataloader)
                train_metrics = train_metrics + self.get_metrics(output, label) / len(train_dataloader)
    
    
            with torch.no_grad():
                val_accuracy = val_loss = val_precision = 0.
                val_recall = val_f1 = val_auroc = 0.
                
                val_metrics = np.array([val_accuracy, val_precision, val_recall, val_f1, val_auroc])
                
                for data, label in val_dataloader:
                    data = data.to(self.device)
                    label = label.to(self.device)
    
                    val_output = self.model(data)[:,0]              
                    val_loss = self.criterion(val_output, label.float())
            
                    val_loss += val_loss / len(val_dataloader)
                    val_metrics = val_metrics + self.get_metrics(val_output, label) / len(val_dataloader)
                    
            curr_lr = self.optimizer.param_groups[0]['lr']

            if log:
                wandb.log({"acc": train_metrics[0], "val_acc": val_metrics[0],
                           "loss": loss,"val_loss":val_loss,"lr":curr_lr,
                           "precision": train_metrics[1], "val_precision": val_metrics[1],
                           "recall":train_metrics[2], "val_recall":val_metrics[2],
                           "f1":train_metrics[3], "val_f1":val_metrics[3],
                           "auroc": train_metrics[4], "val_auroc": val_metrics[4]})

            print(f"lr = {curr_lr}")
            print(f"Epoch : {epoch+1} - loss : {loss:.4f} - acc: {train_metrics[0]:.4f} - val_loss : {val_loss:.4f} - val_acc: {val_metrics[0]:.4f}\n")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\nSaving best model for epoch: {epoch+1}\n")
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.criterion,
                    }, r'out\best_model.pt')

            self.scheduler.step(val_loss)   

        self.model.load_state_dict(torch.load(r'out\best_model.pt')['model_state_dict'])
        

    
    def predict(self, dl):
        '''
        Predicts labels. Should be called after fitting
        '''
        out = []
        y_true = []

        with torch.no_grad():
            for data, label in dl:
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model(data)[:,0]
                out.extend(output.cpu().numpy())
                y_true.extend(label.cpu().numpy())
        
        return out, y_true
    
    
    
    def evaluate(self, test_dataloader):
        '''
        Predicts labels and calculates metrics
        '''
        y_pred = []
        y_true = []
        out = []

        self.model.eval()

        with torch.no_grad():
            accuracy = test_loss = precision = 0.
            recall = f1 = auroc = 0.
            
            test_metrics = np.array([accuracy, precision, recall, f1, auroc])
            
            for data, label in test_dataloader:
                data = data.to(self.device)
                label = label.to(self.device)

                test_output = self.model(data)[:,0]
                test_loss = self.criterion(test_output, label.float())

                test_loss += test_loss.float() / len(test_dataloader)
                test_metrics = test_metrics + self.get_metrics(test_output, label) / len(test_dataloader)

                out.extend(test_output.cpu().numpy())
                y_pred.extend(test_output.round().cpu().numpy())
                y_true.extend(label.cpu().numpy())

        print(f"Loss: {test_loss}")
        print(f"Accuracy: {test_metrics[0]}")
        print(f"Precision: {test_metrics[1]}")
        print(f"Recall: {test_metrics[2]}")
        print(f"F1 Score: {test_metrics[3]}")
        print(f"Area under ROC curve: {test_metrics[4]}\n")

        metrics = {'loss': test_loss,
                   'acc': test_metrics[0],
                   'precision': test_metrics[1],
                   'recall': test_metrics[2],
                   'f1': test_metrics[3],
                   'auroc': test_metrics[4]}

        return metrics