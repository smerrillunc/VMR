import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=512, num_heads=8, num_layers=2, max_seq_len=50):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)  # Project input to embedding dim
        self.pos_encoder = self._generate_sinusoidal_positional_encoding(max_seq_len, embed_dim)
        # can't batch first because we require a mask
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=False), num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)  # Project to final embedding

    def _generate_sinusoidal_positional_encoding(self, max_len, embed_dim):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)

    def forward(self, x, mask):
        x = self.input_proj(x)  # Shape: (seq_len, embed_dim)
        seq_len = x.size(0)
        x = x + self.pos_encoder[:, :seq_len, :].squeeze(0).to(x.device)
        x = self.transformer(x.unsqueeze(1), src_key_padding_mask=mask).squeeze(1)
        x = x.mean(dim=0)  # Aggregate sequence to fixed-size embedding
        return self.output_proj(x)
