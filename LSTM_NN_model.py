import torch
import torch.nn as nn

    

class TrajectoryPredictorLSTM(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        
        self.encoder = nn.LSTM(
            input_size=2,  # x, y coordinates
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x,):
        # x shape: [batch, 11, 2]
        batch_size = x.size(0)
        
        # Encode history
        hidden, state = self.encoder(x)
        
        # Initialize decoder input as last position
        decoder_input = x[:, -1:, :]  # shape: [batch, 1, 2]
        
        # Collect outputs
        outputs = []
        
        # Generate future trajectory
        for t in range(80):  # 80 future timesteps
            # Get decoder output
            out, state = self.decoder(decoder_input, state)
            prediction = self.fc(out)
            outputs.append(prediction)
            
            # Next input is prediction
            decoder_input = prediction
            
        return torch.cat(outputs, dim=1)
