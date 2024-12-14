import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=32, freeze_backbone=False):
        super(ViTEncoder, self).__init__()

        if pretrained:
          self.vit = ViTForImageClassification.from_pretrained("nateraw/food")
        else:
          self.vit = vit_b_16(weights=None)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.vit.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.vit(x).logits