import torch
import torch.nn as nn
from .time_series import TimeSeriesTransformer
from .text import TextTransformer
from .resnet import ResNet

class MultiModalModelR(nn.Module):
    def __init__(self, cgm_seq_len, textual_features, cnn_pretrained=True, cnn_num_blocks=None, cnn_freeze_backbone=False, cnn_output_dim=32):
        super(MultiModalModel, self).__init__()
        self.time_series_transformer = TimeSeriesTransformer(input_dim=1, model_dim=64, num_heads=8, num_layers=3, ff_dim=128, dropout=0.1)
        self.fc_textual = TextTransformer(textual_features, 16)
        self.cnn_bf = ResNet(pretrained=cnn_pretrained, num_blocks=cnn_num_blocks, freeze_backbone=cnn_freeze_backbone, output_dim=cnn_output_dim)
        self.cnn_ln = ResNet(pretrained=cnn_pretrained, num_blocks=cnn_num_blocks, freeze_backbone=cnn_freeze_backbone, output_dim=cnn_output_dim)
        self.fc_combined = nn.Sequential(nn.Linear(32 + 16 + 32 + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, cgm_seq, textual, img_bf, img_ln):
        transformer_out = self.time_series_transformer(cgm_seq.unsqueeze(-1))
        if textual.dim() == 2:  # If textual is (batch_size, feature_dim)
            textual = textual.unsqueeze(1)  # Add seq_len=1 dimension
        textual_out = self.fc_textual(textual)
        img_bf_out = self.cnn_bf(img_bf)
        img_ln_out = self.cnn_ln(img_ln)
        combined = torch.cat((transformer_out, textual_out, img_bf_out, img_ln_out), dim=1)
        output = self.fc_combined(combined)
        return output