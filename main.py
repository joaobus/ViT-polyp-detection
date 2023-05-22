import os
import torch
import numpy as np
import pandas as pd

from model.vit import VisionTransformer
from model.comparison import vgg16, resnet50, resnet101
from utils.data_utils import DataloadingManager
from utils.eval_utils import plot_confusion_matrix, plot_curves
from utils.configs import get_vit_config
from train import BaseClassifier

np.random.seed(0)
torch.manual_seed(0)


def train_model(model_name: str = 'vit', dataset: str = 'cp-child', out_dir: str = 'out/model'):
    models = ['vit','vgg16','resnet50','resnet101']
    assert model_name in models, f'model_name should be one of {models}'

    model_configs = get_vit_config(get_test_model=False)

    if model_name == 'vit':
        model = VisionTransformer(in_channels = model_configs['c'],
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
    elif model_name == 'vgg16':
        model = vgg16()
    elif model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'resnet101':
        model = resnet101()
    
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
    # train_model(model_name='vit', dataset='cp-child-specnorm', out_dir='out/ViT_specnorm')
    # train_model(model_name='vit', dataset='cp-child', out_dir='out/ViT_no_preprocessing')

    # train_model(model_name='resnet50', dataset='cp-child-specnorm', out_dir='out/ResNet50_specnorm')
    # train_model(model_name='resnet50', dataset='cp-child', out_dir='out/ResNet50_no_preprocessing')

    train_model(model_name='resnet101', dataset='cp-child-specnorm', out_dir='out/ResNet101_specnorm')
    train_model(model_name='resnet101', dataset='cp-child', out_dir='out/ResNet101_no_preprocessing')
    train_model(model_name='vgg16', dataset='cp-child-specnorm', out_dir='out/VGG16_specnorm')
    train_model(model_name='vgg16', dataset='cp-child', out_dir='out/VGG16_no_preprocessing')