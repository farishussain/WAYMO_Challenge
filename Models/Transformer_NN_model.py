import torch
import torch.nn as nn
from torch.nn import Transformer

class TrajectoryPredictorTransformer(nn.Module):
    def __init__(self, input_dim=2, seq_len=11, pred_len=80, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer = Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, input_dim)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, src):
        # Encode historical trajectories
        src = self.encoder(src)  # [batch_size, seq_len, d_model]
        
        # Add positional encodings (optional for temporal data)
        # Process with Transformer
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        tgt = torch.zeros(self.pred_len, src.size(1), src.size(2), device=src.device)  # Future input placeholder
        output = self.transformer(src, tgt)

        # Decode predictions
        return self.decoder(output.permute(1, 0, 2))  # [batch_size, pred_len, input_dim]
    