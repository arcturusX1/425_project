"""
Conditional VAE (CVAE) implementation for genre-conditioned music feature learning.
The CVAE learns to generate features conditioned on genre labels, enabling
disentangled representations where genre information is explicitly modeled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, input_dim=40, latent_dim=10, n_classes=10, embedding_dim=8):
        """
        Conditional VAE that conditions on genre labels.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            n_classes: Number of genre classes
            embedding_dim: Dimension of genre embedding
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        # Genre embedding layer
        self.genre_embedding = nn.Embedding(n_classes, embedding_dim)
        
        # Encoder: input + genre -> latent
        self.fc1 = nn.Linear(input_dim + embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: latent + genre -> reconstruction
        self.fc3 = nn.Linear(latent_dim + embedding_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, input_dim)
    
    def encode(self, x, y):
        """
        Encode input x conditioned on genre y.
        
        Args:
            x: Input features [batch_size, input_dim]
            y: Genre labels [batch_size]
        """
        # Embed genre labels
        y_emb = self.genre_embedding(y)  # [batch_size, embedding_dim]
        
        # Concatenate input with genre embedding
        x_cond = torch.cat([x, y_emb], dim=1)  # [batch_size, input_dim + embedding_dim]
        
        h = F.relu(self.fc1(x_cond))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        """
        Decode latent z conditioned on genre y.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            y: Genre labels [batch_size]
        """
        # Embed genre labels
        y_emb = self.genre_embedding(y)  # [batch_size, embedding_dim]
        
        # Concatenate latent with genre embedding
        z_cond = torch.cat([z, y_emb], dim=1)  # [batch_size, latent_dim + embedding_dim]
        
        h = F.relu(self.fc3(z_cond))
        h = F.relu(self.fc4(h))
        return self.fc_out(h)
    
    def forward(self, x, y):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            y: Genre labels [batch_size]
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


class BetaVAE(nn.Module):
    """
    Beta-VAE for disentangled representation learning.
    Beta parameter controls the trade-off between reconstruction quality
    and disentanglement (higher beta = more disentanglement).
    """
    def __init__(self, input_dim=40, latent_dim=10, beta=4.0):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            beta: Disentanglement factor (beta > 1 encourages disentanglement)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc_out(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

