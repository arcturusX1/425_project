"""
Training script for Conditional VAE (CVAE) and Beta-VAE.
Supports both architectures for disentangled representation learning.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from cvae import CVAE, BetaVAE


def loss_function_cvae(recon_x, x, mu, logvar, beta=1.0):
    """Loss function for CVAE: reconstruction + KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld) / x.size(0)


def loss_function_betavae(recon_x, x, mu, logvar, beta=4.0):
    """Loss function for Beta-VAE: reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld) / x.size(0)


def train(model, loader, optimizer, device, model_type='cvae', beta=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if model_type == 'cvae':
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, y)
            loss = loss_function_cvae(recon, x, mu, logvar, beta=beta)
        else:  # betavae
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_function_betavae(recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device, model_type='cvae', beta=1.0):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if model_type == 'cvae':
                x, y = batch[0].to(device), batch[1].to(device)
                recon, mu, logvar = model(x, y)
                loss = loss_function_cvae(recon, x, mu, logvar, beta=beta)
            else:  # betavae
                x = batch[0].to(device)
                recon, mu, logvar = model(x)
                loss = loss_function_betavae(recon, x, mu, logvar, beta=beta)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Get project root directory (assuming script is in src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    X_path = os.path.join(project_root, 'data', 'processed', 'X.npy')
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Feature data not found at {X_path}. Run dataset.py first.")
    
    X = np.load(X_path)
    y_path = os.path.join(project_root, 'data', 'processed', 'y.npy')
    y = np.load(y_path)
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    
    # Create dataset
    if args.model_type == 'cvae':
        dataset = TensorDataset(X, y)
    else:  # betavae
        dataset = TensorDataset(X)
    
    # Train/validation split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    torch.manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.model_type == 'cvae':
        n_classes = len(np.unique(y.numpy()))
        model = CVAE(
            input_dim=X.shape[1],
            latent_dim=args.latent_dim,
            n_classes=n_classes,
            embedding_dim=args.embedding_dim
        ).to(device)
    else:  # betavae
        model = BetaVAE(
            input_dim=X.shape[1],
            latent_dim=args.latent_dim,
            beta=args.beta
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    best_val = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, args.model_type, args.beta)
        val_loss = validate(model, val_loader, device, args.model_type, args.beta)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(results_dir, args.checkpoint)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation improved; saved checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break
    
    # Save final model
    final_path = os.path.join(results_dir, f'{args.model_type}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Encode latent representations
    ckpt_path = os.path.join(results_dir, args.checkpoint)
    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint for encoding: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    model.eval()
    with torch.no_grad():
        X_all = X.to(device)
        if args.model_type == 'cvae':
            y_all = y.to(device)
            mu, logvar = model.encode(X_all, y_all)
        else:  # betavae
            mu, logvar = model.encode(X_all)
        Z = mu.cpu().numpy()
    
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    z_path = os.path.join(processed_dir, f'Z_{args.model_type}.npy')
    np.save(z_path, Z)
    print(f"Saved {args.model_type} latents to {z_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='cvae', choices=['cvae', 'betavae'],
                        help='Model type: cvae (Conditional VAE) or betavae (Beta-VAE)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--embedding-dim', type=int, default=8,
                        help='Genre embedding dimension (for CVAE only)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta parameter for Beta-VAE (disentanglement factor)')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='cvae_best.pth')
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    
    # Set default checkpoint name based on model type
    if args.checkpoint == 'cvae_best.pth' and args.model_type == 'betavae':
        args.checkpoint = 'betavae_best.pth'
    
    main(args)

