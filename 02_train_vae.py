#python 02_train_vae.py --outdir toy_circle_data --train_types 0123 --warmup_ae_epochs 80 --vae_epochs 120 --beta_max 5e-4

import os
import json
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def chunk_iter(N, bs):
    for i in range(0, N, bs):
        yield slice(i, min(i+bs, N))


# =========================
# A more expressive MLP block
# =========================
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class ResMLP(nn.Module):
    """
    Residual MLP: x -> proj -> (res blocks) -> out
    """
    def __init__(self, d_in, d_hidden, n_layers=4, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([MLPBlock(d_hidden, d_hidden, dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_hidden)

    def forward(self, x):
        h = self.in_proj(x)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(ln(h))
        h = self.out_norm(h)
        return h


# =========================
# VAE: encoder/decoder
# =========================
class VAE(nn.Module):
    def __init__(self, L: int, latent_dim: int, hidden: int = 512,
                 enc_layers: int = 6, dec_layers: int = 6, dropout: float = 0.05):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim

        self.encoder = ResMLP(L, hidden, n_layers=enc_layers, dropout=dropout)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            *[nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
              for _ in range(dec_layers)],
            nn.Linear(hidden, L),
            nn.Tanh()  # output in [-1,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)


# =========================
# Loss: recon + beta*KL with optional free-bits
# =========================
def kl_diag_gaussian(mu, logvar):
    # KL(q||p) for q=N(mu, diag(exp(logvar))), p=N(0,I)
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)  # (B,)

def vae_loss(x, x_hat, mu, logvar, beta: float, free_bits: float = 0.0):
    recon = torch.mean((x - x_hat) ** 2)  # scalar

    kl_per = kl_diag_gaussian(mu, logvar)  # (B,)
    if free_bits > 0.0:
        # "free bits" per latent dim: allow some KL before penalizing
        fb = free_bits * mu.shape[1]
        kl_per = torch.clamp(kl_per, min=fb)
    kl = torch.mean(kl_per)

    return recon + beta * kl, recon.detach(), kl.detach()


# =========================
# Training
# =========================
@dataclass
class TrainCfg:
    outdir: str
    seed: int = 0
    latent_dim: int = 2
    hidden: int = 512
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.05

    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # staged schedule
    warmup_ae_epochs: int = 80     # beta=0, deterministic z=mu
    vae_epochs: int = 120          # KL anneal
    beta_max: float = 5e-4         # tiny KL for 2D latent
    free_bits: float = 0.0         # set to 0.1 if needed

    train_types: str = "01"        # "01" -> use types {0,1}. "0123" -> all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--latent_dim", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.05)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--warmup_ae_epochs", type=int, default=80)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--beta_max", type=float, default=5e-4)
    ap.add_argument("--free_bits", type=float, default=0.0)

    ap.add_argument("--train_types", type=str, default="01",
                    help="Which type labels to train on, e.g. '01' or '0123'")
    args = ap.parse_args()

    cfg = TrainCfg(
        outdir=args.outdir,
        seed=args.seed,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ae_epochs=args.warmup_ae_epochs,
        vae_epochs=args.vae_epochs,
        beta_max=args.beta_max,
        free_bits=args.free_bits,
        train_types=args.train_types,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- Load data
    data_npz = os.path.join(cfg.outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run 00_generate_data.py first."

    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)      # radians (N,L)
    y = data["y"].astype(np.float32)
    types = data["types"].astype(np.int64)

    N, L = X.shape
    print(f"Loaded dataset: N={N}, L={L}. Oracle: min={y.min():.3f} max={y.max():.3f}")

    # ---- Filter training set
    allowed = sorted({int(c) for c in cfg.train_types})
    mask = np.isin(types, allowed)
    Xtr = X[mask]
    ytr = y[mask]
    print(f"Training on types={allowed}  -> {len(Xtr)} samples. Oracle max in train={float(ytr.max()):.3f}")

    # Scale for network: [-pi,pi] -> [-1,1]
    Xtr_scaled = (Xtr / np.pi).astype(np.float32)

    # ---- Model
    vae = VAE(
        L=L,
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        dropout=cfg.dropout
    ).to(device)

    opt = optim.AdamW(vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- Output dirs
    model_dir = os.path.join(cfg.outdir, "models")
    plot_dir = os.path.join(cfg.outdir, "plots_vae")
    ensure_dir(model_dir)
    ensure_dir(plot_dir)

    # ---- Training loops
    X_t = torch.from_numpy(Xtr_scaled).to(device)
    n_train = X_t.shape[0]

    hist = {"loss": [], "recon": [], "kl": [], "beta": []}

    def run_epoch(beta: float, deterministic: bool):
        vae.train()
        perm = torch.randperm(n_train, device=device)
        tot_loss = tot_recon = tot_kl = 0.0

        for sl in chunk_iter(n_train, cfg.batch_size):
            idx = perm[sl]
            xb = X_t[idx]

            opt.zero_grad(set_to_none=True)
            mu, logvar = vae.encode(xb)

            if deterministic:
                z = mu
            else:
                z = vae.reparam(mu, logvar)

            x_hat = vae.decode(z)
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=beta, free_bits=cfg.free_bits)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip)

            opt.step()

            bs = xb.shape[0]
            tot_loss += float(loss) * bs
            tot_recon += float(recon) * bs
            tot_kl += float(kl) * bs

        return tot_loss / n_train, tot_recon / n_train, tot_kl / n_train

    # Stage 1: Autoencoder warmup (beta=0, deterministic)
    print("\n[Stage 1] AE warmup (beta=0, deterministic z=mu)")
    for ep in range(1, cfg.warmup_ae_epochs + 1):
        loss, recon, kl = run_epoch(beta=0.0, deterministic=True)
        hist["loss"].append(loss); hist["recon"].append(recon); hist["kl"].append(kl); hist["beta"].append(0.0)
        if ep == 1 or ep % 10 == 0:
            print(f"AE ep {ep:3d} | loss={loss:.4f} recon={recon:.4f} kl={kl:.4f}")

    # Stage 2: VAE fine-tuning (KL anneal)
    print("\n[Stage 2] VAE fine-tune (anneal beta, sampled z)")
    for ep in range(1, cfg.vae_epochs + 1):
        beta = cfg.beta_max * min(1.0, ep / max(1, cfg.vae_epochs))
        loss, recon, kl = run_epoch(beta=beta, deterministic=False)
        hist["loss"].append(loss); hist["recon"].append(recon); hist["kl"].append(kl); hist["beta"].append(beta)
        if ep == 1 or ep % 10 == 0:
            print(f"VAE ep {ep:3d} | beta={beta:.6f} | loss={loss:.4f} recon={recon:.4f} kl={kl:.4f}")

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_t[:4000])
        std = torch.exp(0.5 * logvar)

    print("mu std per dim:", mu.std(dim=0).cpu().numpy())
    print("mean std (posterior) per dim:", std.mean(dim=0).cpu().numpy())

    # ---- Save model
    ckpt = {
        "state_dict": vae.state_dict(),
        "L": L,
        "latent_dim": cfg.latent_dim,
        "hidden": cfg.hidden,
        "enc_layers": cfg.enc_layers,
        "dec_layers": cfg.dec_layers,
        "dropout": cfg.dropout,
        "train_types": cfg.train_types,
        "seed": cfg.seed,
    }
    vae_path = os.path.join(model_dir, "vae.pt")
    torch.save(ckpt, vae_path)
    print("\nSaved VAE to:", os.path.abspath(vae_path))

    # ---- Save curves
    plt.figure(figsize=(7,4))
    plt.plot(hist["loss"], label="total")
    plt.plot(hist["recon"], label="recon")
    plt.plot(hist["kl"], label="kl")
    plt.title("Training curves")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7,3))
    plt.plot(hist["beta"])
    plt.title("beta schedule")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "beta_schedule.png"), dpi=160)
    plt.close()

    # ---- Latent scatter (means) on subset
    vae.eval()
    m = min(4000, n_train)
    with torch.no_grad():
        mu, logvar = vae.encode(X_t[:m])
        Z_mu = mu.cpu().numpy()
        Z_s  = vae.reparam(mu, logvar).cpu().numpy()  # sampled z, if you want

    types_sub = types[mask][:m]
    y_sub = ytr[:m]

    # ---- Plot 1: categorical types
    plt.figure(figsize=(6,5))
    for t in np.unique(types_sub):
        sel = (types_sub == t)
        plt.scatter(Z_mu[sel,0], Z_mu[sel,1], s=10, label=f"type {t}", alpha=0.65)

    plt.title("Latent means by type")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latent_types.png"), dpi=160)
    
    plt.figure(figsize=(6,5))
    for t in np.unique(types_sub):
        sel = (types_sub == t)
        plt.scatter(Z_s[sel,0], Z_s[sel,1], s=10, label=f"type {t}", alpha=0.65)

    plt.title("Latent means by type (sampled z)")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latent_types_sampled.png"), dpi=160)

    # ---- Plot 2: continuous oracle
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z_mu[:,0], Z_mu[:,1], c=y_sub, s=10)
    plt.colorbar(sc, label="oracle f(x)")
    plt.title("Latent means colored by oracle")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latent_oracle.png"), dpi=160)
    
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z_s[:,0], Z_s[:,1], c=y_sub, s=10)
    plt.colorbar(sc, label="oracle f(x)")
    plt.title("Latent means colored by oracle (sampled z)")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latent_oracle_sampled.png"), dpi=160)

    # ---- Save train meta
    meta = {
        "train_types": cfg.train_types,
        "N_train": int(n_train),
        "L": int(L),
        "latent_dim": int(cfg.latent_dim),
        "hidden": int(cfg.hidden),
        "enc_layers": int(cfg.enc_layers),
        "dec_layers": int(cfg.dec_layers),
        "dropout": float(cfg.dropout),
        "warmup_ae_epochs": int(cfg.warmup_ae_epochs),
        "vae_epochs": int(cfg.vae_epochs),
        "beta_max": float(cfg.beta_max),
        "free_bits": float(cfg.free_bits),
        "seed": int(cfg.seed),
    }
    with open(os.path.join(plot_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved plots to:", os.path.abspath(plot_dir))


if __name__ == "__main__":
    main()
