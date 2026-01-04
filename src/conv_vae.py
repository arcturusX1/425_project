import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # -> 32 x H/2 x W/2
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> 64 x H/4 x W/4
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> 128 x H/8 x W/8

        # we'll infer flatten size at runtime if needed; assume input 64x128 -> H/8=8, W/8=16 => 128*8*16=16384
        self._flatten_dim = 128 * 8 * 16
        self.fc_mu = nn.Linear(self._flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flatten_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self._flatten_dim)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(h.size(0), 128, 8, 16)
        h = F.relu(self.dec3(h))
        h = F.relu(self.dec2(h))
        x_recon = torch.sigmoid(self.dec1(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
