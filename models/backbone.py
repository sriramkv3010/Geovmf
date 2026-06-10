import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.resnet18(weights=weights)
        # Drop avgpool + fc; keep everything up to the final global avg-pool
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)
