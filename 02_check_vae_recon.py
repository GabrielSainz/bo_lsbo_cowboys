#python 02_check_vae_recon.py --outdir toy_circle_data --n_random 10


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.spatial.distance import cdist


# -------------------------
# Turtle + oracle (same as before)
# -------------------------
def turtle_path(delta_theta: np.ndarray, step_size: float) -> np.ndarray:
    delta_theta = np.asarray(delta_theta).reshape(-1)
    theta = 0.0
    p = np.array([0.0, 0.0], dtype=float)
    pts = [p.copy()]
    for dth in delta_theta:
        theta += float(dth)
        p = p + step_size * np.array([np.cos(theta), np.sin(theta)])
        pts.append(p.copy())
    return np.stack(pts, axis=0)

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    D = cdist(A, B)
    return float(D.min(axis=1).mean() + D.min(axis=0).mean())

def oracle_f(delta_theta, target_pts, step_size, w_close=0.2, w_smooth=0.05):
    pts = turtle_path(delta_theta, step_size=step_size)
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    shape_loss = chamfer_distance(pts0, target_pts)

    L = len(delta_theta)
    closure = np.sum((pts[-1] - pts[0])**2) / ((L * step_size)**2)

    rough = np.mean(np.diff(delta_theta)**2) / (np.pi**2) if L > 1 else 0.0
    return -(shape_loss + w_close * closure + w_smooth * rough)


# -------------------------
# VAE (must match training)
# -------------------------
import torch
import torch.nn as nn

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
        return self.out_norm(h)

class VAE(nn.Module):
    def __init__(self, L: int, latent_dim: int, hidden: int = 512,
                 enc_layers: int = 6, dec_layers: int = 6, dropout: float = 0.05):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim

        self.encoder = ResMLP(L, hidden, n_layers=enc_layers, dropout=dropout)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        dec_blocks = []
        for _ in range(dec_layers):
            dec_blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden),
            ))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            *dec_blocks,
            nn.Linear(hidden, L),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(z)

def load_vae(vae_path: str, device):
    ckpt = torch.load(vae_path, map_location=device)

    L = ckpt["L"]
    latent_dim = ckpt["latent_dim"]
    hidden = ckpt.get("hidden", 512)
    enc_layers = ckpt.get("enc_layers", 6)
    dec_layers = ckpt.get("dec_layers", 6)
    dropout = ckpt.get("dropout", 0.05)

    vae = VAE(L=L, latent_dim=latent_dim, hidden=hidden,
              enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout).to(device)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.eval()
    return vae, L, latent_dim, hidden



def reconstruct_x(vae, x_radians: np.ndarray, device):
    """
    x_radians: (L,) in [-pi, pi]
    returns:
      x_hat_radians: (L,)
      z_mu: (latent_dim,)
    """
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device).view(1, -1)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
        x_hat_scaled = vae.decode(mu)
    x_hat = (x_hat_scaled.cpu().numpy().reshape(-1) * np.pi).astype(np.float64)
    z_mu = mu.cpu().numpy().reshape(-1)
    return x_hat, z_mu


# -------------------------
# Plot one example: true vs recon
# -------------------------
def plot_example(x, x_hat, y_true, y_hat, z_mu, target, step_size, outpath, title):
    pts = turtle_path(x, step_size)
    pts_hat = turtle_path(x_hat, step_size)

    fig = plt.figure(figsize=(14, 4.5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(pts[:, 0], pts[:, 1], linewidth=2)
    ax1.scatter([pts[0, 0]], [pts[0, 1]], s=40)
    ax1.set_aspect("equal", "box")
    ax1.axis("off")
    ax1.set_title(f"TRUE path\noracle={y_true:.3f}")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(pts_hat[:, 0], pts_hat[:, 1], linewidth=2)
    ax2.scatter([pts_hat[0, 0]], [pts_hat[0, 1]], s=40)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"RECON path (decode(mu))\noracle={y_hat:.3f}")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x, label="true")
    ax3.plot(x_hat, label="recon", linestyle="--")
    ax3.axhline(0.0, linewidth=1, linestyle=":")
    ax3.set_title(f"Δθ sequence\nz_mu={np.array2string(z_mu, precision=2)}")
    ax3.set_xlabel("t")
    ax3.set_ylabel("Δθ (radians)")
    ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="toy_circle_data")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--n_random", type=int, default=2)
    args = parser.parse_args()

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    vae_path = args.vae_path or os.path.join(outdir, "models", "vae.pt")

    # load dataset
    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)        # radians
    y = data["y"].astype(np.float64)
    target = data["target"].astype(np.float64)

    # load config for step_size (fallback 1.0)
    cfg_path = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        step_size = float(cfg.get("step_size", 1.0))
        w_close = float(cfg.get("w_close", 0.2))
        w_smooth = float(cfg.get("w_smooth", 0.05))
    else:
        step_size = 1.0
        w_close = 0.2
        w_smooth = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load VAE
    assert os.path.exists(vae_path), f"VAE not found at {vae_path}"
    vae, L, latent_dim, hidden = load_vae(vae_path, device)
    print(f"Loaded VAE: L={L}, latent_dim={latent_dim}, hidden={hidden}")

    # choose indices: best, median, worst, random
    best_idx = int(np.argmax(y))
    worst_idx = int(np.argmin(y))
    med_idx = int(np.argsort(y)[len(y)//2])

    rng = np.random.default_rng(0)
    rand_idx = rng.choice(len(X), size=args.n_random, replace=False).tolist()

    idx_list = [("best", best_idx), ("median", med_idx), ("worst", worst_idx)]
    idx_list += [(f"rand{i+1}", ix) for i, ix in enumerate(rand_idx)]

    # output folder
    recon_dir = os.path.join(outdir, "recon_checks")
    os.makedirs(recon_dir, exist_ok=True)

    print("\nReconstruction report:")
    for tag, idx in idx_list:
        x = X[idx]
        x_hat, z_mu = reconstruct_x(vae, x, device)

        # metrics in sequence space
        mse = float(np.mean((x - x_hat) ** 2))
        mean_abs_true = float(np.mean(np.abs(x)))
        mean_abs_hat = float(np.mean(np.abs(x_hat)))
        max_abs_true = float(np.max(np.abs(x)))
        max_abs_hat = float(np.max(np.abs(x_hat)))

        # oracle check (true vs recon)
        y_true = float(oracle_f(x, target, step_size, w_close, w_smooth))
        y_hat = float(oracle_f(x_hat, target, step_size, w_close, w_smooth))

        print(f"- {tag:6s} idx={idx:6d} | mse={mse: .4f} | "
              f"mean|Δθ| true={mean_abs_true: .3f} recon={mean_abs_hat: .3f} | "
              f"max|Δθ| true={max_abs_true: .3f} recon={max_abs_hat: .3f} | "
              f"oracle true={y_true: .3f} recon={y_hat: .3f}")

        outpath = os.path.join(recon_dir, f"recon_{tag}.png")
        plot_example(
            x=x,
            x_hat=x_hat,
            y_true=y_true,
            y_hat=y_hat,
            z_mu=z_mu,
            target=target,
            step_size=step_size,
            outpath=outpath,
            title=f"Reconstruction check: {tag} (idx={idx})"
        )

    print(f"\nSaved recon plots to: {os.path.abspath(recon_dir)}")
    print("Open images: recon_best.png, recon_median.png, recon_worst.png, recon_rand*.png")


if __name__ == "__main__":
    main()
