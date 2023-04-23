import torch.nn as nn
import torchvision.transforms as transforms

class DataAugmentation(nn.Module):
    def __init__(self,img_size: int = 224):
        super().__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size,scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

    def forward(self, x):
        x = self.transforms(x)
        return x