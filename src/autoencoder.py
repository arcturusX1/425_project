"""
Standard Autoencoder (non-variational) for comparison with VAE methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim=40, latent_dim=10):
        """
        Standard autoencoder (non-variational).
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_latent = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_latent(h)
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc_out(h)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

