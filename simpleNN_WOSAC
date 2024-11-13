import torch
import torch.nn as nn

class SimpleNNWOSAC(nn.Module):
    def __init__(self, history_size=10, hidden_size=64, future_size=5):
        """
        Simple trajectory prediction network with combined encoder-decoder.

        Args:
            history_size (int): Number of historical timesteps
            hidden_size (int): Size of hidden layers
            future_size (int): Number of future timesteps to predict
        """
        super().__init__()

        # Define a sequential model with an additional hidden layer
        self.net = nn.Sequential(
            # Input layer that maps the history (10 timesteps * 2 coordinates) to the hidden size
            nn.Linear(history_size * 2, hidden_size),
            nn.ReLU(),
            # First hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Additional hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output layer that maps the hidden representation to the future trajectory (5 timesteps * 2 coordinates)
            nn.Linear(hidden_size, future_size * 2)
        )
    
    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, history_size * 2]
                            containing historical (x,y) positions

        Returns:
            torch.Tensor: Predicted future trajectories of shape [batch_size, future_size * 2]
        """
        # Pass the input tensor through the combined encoder-decoder network
        predictions = self.net(x)
        return predictions

# Example usage
def main():
    # Create model
    model = SimpleNNWOSAC(history_size=10, hidden_size=64, future_size=5)
    
    # Create sample data
    batch_size = 32
    history_data = torch.randn(batch_size, 20)  # 10 timesteps * 2 (x,y)
    
    # Forward pass
    predictions = model(history_data)
    
    print(f"Input shape: {history_data.shape}")
    print(f"Output shape: {predictions.shape}")
    
    return model, predictions

if __name__ == "__main__":
    model, predictions = main()
