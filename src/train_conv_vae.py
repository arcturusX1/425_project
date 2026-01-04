import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from conv_vae import ConvVAE


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kld) / x.size(0)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            loss = loss_function(recon, x, mu, logvar)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    data_path = os.path.join('data', 'processed', 'X_spec.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Spectrogram data not found at {data_path}. Run dataset_spectrogram.py first.")

    X = np.load(data_path)
    X = torch.from_numpy(X).float()

    dataset = TensorDataset(X)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    torch.manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ConvVAE(in_channels=X.shape[1], latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs('results', exist_ok=True)
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join('results', args.checkpoint)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation improved; saved checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break

    final_path = os.path.join('results', 'conv_vae_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # encode latent representations
    ckpt_path = os.path.join('results', args.checkpoint)
    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint for encoding: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print('Best checkpoint not found; using current model to encode latents.')

    model.eval()
    with torch.no_grad():
        X_all = X.to(device)
        mu, logvar = model.encode(X_all)
        Z = mu.cpu().numpy()

    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    z_path = os.path.join('data', 'processed', 'Z_conv.npy')
    np.save(z_path, Z)
    print(f"Saved conv-VAE latents to {z_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='conv_vae_best.pth')
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    main(args)
