import torch.nn as nn
import torchvision.models as models
from utils.configs import get_vit_config


class ComparisonModel(nn.Module):
    def __init__(self, 
                 base_model, 
                 n_classes: int = 1, 
                 drop_p: float = 0.):
        super().__init__()
        self.base = base_model
        self.out = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(1000,n_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.base(x)
        return self.out(x)


# Models
def vgg16():
    return ComparisonModel(models.vgg16_bn(), drop_p=0.6) 

def resnet50():
    return ComparisonModel(models.resnet50(), drop_p=0.6)

def resnet101():
    return ComparisonModel(models.resnet101(), drop_p=0.6)