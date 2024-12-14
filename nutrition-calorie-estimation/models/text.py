import torch
import torch.nn as nn

class TextTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, dim_feedforward=128):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, dim_feedforward))
        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation="relu",
        )
        self.fc_out = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer(x, x)
        return self.fc_out(x.mean(dim=1))