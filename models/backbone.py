"""
models/backbone.py
==================
ResNet-18 feature extractor — strips the final classification head and
returns a 512-dimensional embedding per image.

Used as the shared visual backbone for all downstream modules
(GPM, IP, SOT, vMF head).
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 backbone with the classification head removed.

    Forward pass: image (B, 3, 224, 224) → features (B, 512).

    Parameters
    ----------
    pretrained : bool
        If ``True``, loads ImageNet-1K weights (IMAGENET1K_V1).
        If ``False``, initialises with random weights.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.resnet18(weights=weights)
        # Drop avgpool + fc; keep everything up to the final global avg-pool
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, H, W)

        Returns
        -------
        features : torch.Tensor, shape (B, 512)
        """
        return self.features(x).view(x.size(0), -1)
