import torch
import torch.nn as nn
from .time_series import TimeSeriesTransformer
from .text import TextTransformer
from .vision import ViTEncoder

class MultiModalModel(nn.Module):
    def __init__(self, cgm_seq_len, textual_features, vit_pretrained=True, vit_output_dim=32, vit_freeze_backbone=False):
        super(MultiModalModel, self).__init__()
        self.time_series_transformer = TimeSeriesTransformer(input_dim=1, model_dim=64, num_heads=8, num_layers=3, ff_dim=128, dropout=0.1)

        self.fc_textual = TextTransformer(textual_features, 16)

        self.vit_bf = ViTEncoder(pretrained=vit_pretrained, output_dim=vit_output_dim, freeze_backbone=vit_freeze_backbone)
        self.vit_ln = ViTEncoder(pretrained=vit_pretrained, output_dim=vit_output_dim, freeze_backbone=vit_freeze_backbone)
        self.fc_combined = nn.Sequential(
            nn.Linear(32 + 16 + vit_output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, cgm_seq, textual, img_bf, img_ln):
        transformer_out = self.time_series_transformer(cgm_seq.unsqueeze(-1))

        if textual.dim() == 2:
           textual = textual.unsqueeze(1)
        textual_out = self.fc_textual(textual)

        img_bf_out = self.vit_bf(img_bf)
        img_ln_out = self.vit_ln(img_ln)

        combined = torch.cat((transformer_out, textual_out, img_bf_out, img_ln_out), dim=1)
        output = self.fc_combined(combined)
        return output