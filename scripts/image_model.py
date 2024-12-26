import torch
import torch.nn as nn
import timm


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, model_name: str = 'efficientnet_b0', pretrained: bool = False) -> None:
        super(EfficientNetBinaryClassifier, self).__init__()
        self.efficientnet = timm.create_model(model_name, pretrained)
        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()
        self.custom_head = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet(x)
        x = self.custom_head(x)
        return x