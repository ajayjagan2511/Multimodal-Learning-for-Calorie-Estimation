import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, pretrained=True, num_blocks=None, freeze_backbone=False, output_dim=32):

        super(ResNet, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        if num_blocks:
            self.features = nn.Sequential(*list(base_model.children())[:num_blocks])
        else:
            self.features = nn.Sequential(*list(base_model.children())[:-1])

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        self.fc = None
        self.output_dim = output_dim

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.output_dim).to(x.device)

        x = self.fc(x)
        return x