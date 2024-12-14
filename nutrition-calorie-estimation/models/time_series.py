import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, model_dim))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim // 2)

    def forward(self, x):
        x = self.embedding(x)
        seq_length = x.size(1)
        x += self.positional_encoding[:seq_length, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.fc_out(x[:, -1, :])
        return x