import torch
from torch import nn




class TrajectoryPredictorGRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_size,      # x, y coordinates
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: [batch, 11, 2]
        batch_size = x.size(0)
        
        # Encode history
        _, hidden = self.encoder(x)
        
        # Initialize decoder input as last position
        decoder_input = x[:, -1:, :]  # shape: [batch, 1, 2]
        
        # Collect outputs
        outputs = []
        
        # Generate future trajectory
        for _ in range(80):  # 80 future timesteps
            # Get decoder output
            output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(output)
            outputs.append(prediction)
            
            # Next input is prediction
            decoder_input = prediction
        
        return torch.cat(outputs, dim=1)

# Architecture breakdown:
# Input: [batch_size, 11, 2]       # 11 timesteps of x,y coordinates
# Encoder GRU: Processes input sequence into hidden states
# Decoder GRU: Generates future positions using hidden state
# Output: [batch_size, 80, 2]      # 80 future timesteps of x,y coordinates