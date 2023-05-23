import torch.nn as nn
import torchvision.models as models


# class ComparisonModel(nn.Module):
#     def __init__(self, 
#                  base_model, 
#                  n_classes: int = 1):
#         super().__init__()
#         base_model.fc = nn.Linear(base_model.fc.in_features, n_classes)
#         self.base = base_model
#         self.activation = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.base(x)
#         return self.activation(x)


# # Models
# def resnet18():
#     return ComparisonModel(models.resnet18())

# def resnet34():
#     return ComparisonModel(models.resnet34())


def build_model(base):
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, 1)
    model = nn.Sequential(
        base,
        nn.Sigmoid()
    )
    return model


def resnet18():
    return build_model(models.resnet18())

def resnet34():
    return build_model(models.resnet34())

