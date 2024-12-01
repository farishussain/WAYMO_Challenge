import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        
        valid_mask = data['agent_valid'].astype(bool)
        
        # Store original data
        self.history = torch.FloatTensor(data['history/xy'][valid_mask])
        self.future = torch.FloatTensor(data['future/xy'][valid_mask])
        self.history_valid = torch.FloatTensor(data['history/valid'][valid_mask])
        self.future_valid = torch.FloatTensor(data['future/valid'][valid_mask])
        
        # Compute statistics only on valid points
        valid_history = self.history[self.history_valid.bool()]
        valid_future = self.future[self.future_valid.bool()]
        
        # Compute mean and std across all valid points
        all_valid_points = torch.cat([valid_history, valid_future], dim=0)
        self.mean = all_valid_points.mean(dim=0)
        self.std = all_valid_points.std(dim=0)
        
        # Add small epsilon to avoid division by zero
        self.std = torch.where(self.std < 1e-6, torch.ones_like(self.std), self.std)
        
        # Normalize
        self.history_normalized = (self.history - self.mean) / self.std
        self.future_normalized = (self.future - self.mean) / self.std
        
        print("Normalization stats:")
        print(f"Mean: {self.mean}")
        print(f"Std: {self.std}")
        print(f"History range before norm: {self.history.min():.2f} to {self.history.max():.2f}")
        print(f"History range after norm: {self.history_normalized.min():.2f} to {self.history_normalized.max():.2f}")

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return ((self.history_normalized[idx], self.history_valid[idx]), 
                (self.future_normalized[idx], self.future_valid[idx]))

    def denormalize(self, data):
        """Denormalize data back to original scale"""
        return data * self.std + self.mean
    
    