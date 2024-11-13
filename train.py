import os
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from simpleNNWOSAC import SimpleNNWOSAC

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, history_size=10, future_size=5):
        """
        Simple dataset that loads trajectory data from numpy files.
        Expects files with shape [n_samples, timesteps, 2]
        """
        self.data = np.load(data_path)
        self.history_size = history_size
        self.future_size = future_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        trajectory = self.data[idx]
        history = trajectory[:self.history_size].reshape(-1)
        future = trajectory[self.history_size:self.history_size + self.future_size].reshape(-1)
        return {
            'history': torch.FloatTensor(history),
            'future': torch.FloatTensor(future)
        }

def train(args):
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Initialize model, optimizer, and loss function
    model = SimpleNNWOSAC(
        history_size=args.history_size,
        hidden_size=args.hidden_size,
        future_size=args.future_size
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Load the last checkpoint if it exists
    last_checkpoint_path = os.path.join(args.save_folder, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Load data
    train_dataset = TrajectoryDataset(
        args.train_data_path,
        history_size=args.history_size,
        future_size=args.future_size
    )
    val_dataset = TrajectoryDataset(
        args.val_data_path,
        history_size=args.history_size,
        future_size=args.future_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.n_epochs):
        # Training phase
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        
        for batch in pbar:
            history = batch['history']
            future = batch['future']
            
            if torch.cuda.is_available():
                history = history.cuda()
                future = future.cuda()
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(history)
            
            # Compute loss and backward pass
            loss = criterion(predictions, future)
            loss.backward()
            
            # Gradient clipping
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()
            train_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch+1}/{args.n_epochs} - Loss: {loss.item():.4f}'
            )
        
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(train_losses),
        }, last_checkpoint_path)
        
        # Validation phase
        if epoch % args.validate_every == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    history = batch['history']
                    future = batch['future']
                    
                    if torch.cuda.is_available():
                        history = history.cuda()
                        future = future.cuda()
                    
                    predictions = model(history)
                    loss = criterion(predictions, future)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(args.save_folder, 'best_model.pth'))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--history_size', type=int, default=10)
    parser.add_argument('--future_size', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--validate_every', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)