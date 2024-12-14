import torch
import torch.nn as nn

import torch

class RMSRELoss(nn.Module):
    def __init__(self):
        super(RMSRELoss, self).__init__()

    def forward(self, predictions, targets):
        error = ((predictions - targets) ** 2) / (targets ** 2 + 1e-8)
        rmsre = torch.sqrt(torch.mean(error))
        return rmsre