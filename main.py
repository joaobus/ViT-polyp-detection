import os
import torch
import numpy as np
import pandas as pd

from model.vit import VisionTransformer
from model.comparison import resnet18, resnet34
from utils.data_utils import DataloadingManager
from utils.eval_utils import plot_confusion_matrix, plot_curves
from utils.configs import get_vit_config
from train import BaseClassifier

np.random.seed(0)
torch.manual_seed(0)


def train_model(model, model_name: str = '-', dataset: str = 'cp-child', out_dir: str = 'out/model'):
    
    train_dataloader, test_dataloader, val_dataloader = DataloadingManager(dataset_name=dataset).get_dataloaders()

    classifier = BaseClassifier(model, model_configs={'architecture':model_name}, get_test_config=False) # type:ignore

    save_path = os.path.join(out_dir, 'best_model.pt')
    # classifier.load_pretrained(save_path)
    classifier.fit(train_dataloader, val_dataloader, log=True, save_model=False, save_path=save_path)

    test_metrics = classifier.evaluate(test_dataloader)
    val_metrics = classifier.evaluate(val_dataloader) 
    train_metrics = classifier.evaluate(train_dataloader)

    metrics = pd.DataFrame({'test':test_metrics, 'train':train_metrics, 'val':val_metrics})
    metrics.to_csv(os.path.join(out_dir, 'metrics.csv'))

    output, y_true = classifier.predict(test_dataloader)

    plot_confusion_matrix(output, y_true, os.path.join(out_dir, 'cf_matrix.png'))
    plot_curves(output, y_true, os.path.join(out_dir, 'curves.png'))  


if __name__ == '__main__':

    model_configs = get_vit_config(get_test_model=False)

    vit = VisionTransformer(in_channels = model_configs['c'],
                        patch_size = model_configs['patch_size'],
                        emb_size = model_configs['emb_size'],
                        img_size = model_configs['h'],
                        drop_p = model_configs['drop_rate'],
                        forward_drop_p= model_configs['drop_rate'],
                        num_heads = model_configs['num_heads'],
                        n_blocks = model_configs['num_blocks'],
                        n_classes = 1,
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        return_attn_weights=model_configs['return_attn_weights'])


    # train_model(model=vit, model_name='ViT',dataset='cp-child-specnorm', out_dir='out/ViT_specnorm')
    # train_model(model=vit, model_name='ViT',dataset='cp-child', out_dir='out/ViT_no_preprocessing')

    # train_model(model=resnet18(), model_name='resnet18',dataset='cp-child-specnorm', out_dir='out/ResNet18_specnorm')
    # train_model(model=resnet18(), model_name='resnet18',dataset='cp-child', out_dir='out/ResNet18_no_preprocessing')

    train_model(model=resnet34(), model_name='resnet34',dataset='cp-child-specnorm', out_dir='out/ResNet34_specnorm')
    train_model(model=resnet34(), model_name='resnet34',dataset='cp-child', out_dir='out/ResNet34_no_preprocessing')