# 08_oracle_heatmap_latent.py
# Build ONE reference heatmap of the TRUE oracle over a latent grid:
#   z  ->  x = decode(z)  ->  y = oracle_f(x)
#
# Usage:
#   python 00_grid_oracle.py --outdir toy_circle_data --z_box 6 --grid_res 400
#
# It will save:
#   <outdir>/oracle_heatmap/z6.0_res140/oracle_heatmap.png

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


# -------------------------
# Utils
# -------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Turtle + oracle (must match your generator)
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

def oracle_f(delta_theta: np.ndarray,
             target_pts: np.ndarray,
             step_size: float,
             w_close: float = 0.2,
             w_smooth: float = 0.05) -> float:
    pts = turtle_path(delta_theta, step_size=step_size)
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    shape_loss = chamfer_distance(pts0, target_pts)

    L = len(delta_theta)
    closure = np.sum((pts[-1] - pts[0])**2) / ((L * step_size)**2)

    rough = np.mean(np.diff(delta_theta)**2) / (np.pi**2) if L > 1 else 0.0
    return -(shape_loss + w_close * closure + w_smooth * rough)


# -------------------------
# Faster batch oracle (same definition, KDTree for chamfer)
# -------------------------
def turtle_path_batch(delta_theta: np.ndarray, step_size: float) -> np.ndarray:
    """
    delta_theta: (N,L) -> pts: (N,L+1,2)
    """
    dt = np.asarray(delta_theta, dtype=np.float64)
    N, L = dt.shape
    theta = np.cumsum(dt, axis=1)  # (N,L)
    inc = step_size * np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (N,L,2)
    pos = np.cumsum(inc, axis=1)  # (N,L,2)
    pts0 = np.zeros((N, 1, 2), dtype=np.float64)
    return np.concatenate([pts0, pos], axis=1)

def chamfer_distance_kdtree(A: np.ndarray, B_tree: cKDTree, B: np.ndarray) -> float:
    d1 = B_tree.query(A, k=1)[0].mean()
    A_tree = cKDTree(A)
    d2 = A_tree.query(B, k=1)[0].mean()
    return float(d1 + d2)

def oracle_f_batch(delta_theta_batch: np.ndarray,
                   target_pts: np.ndarray,
                   step_size: float,
                   w_close: float = 0.2,
                   w_smooth: float = 0.05) -> np.ndarray:
    X = np.asarray(delta_theta_batch, dtype=np.float64)
    N, L = X.shape

    pts = turtle_path_batch(X, step_size=step_size)                # (N,L+1,2)
    pts0 = pts - pts.mean(axis=1, keepdims=True)

    closure = np.sum((pts[:, -1, :] - pts[:, 0, :])**2, axis=1) / ((L * step_size)**2)
    if L > 1:
        rough = np.mean(np.diff(X, axis=1)**2, axis=1) / (np.pi**2)
    else:
        rough = np.zeros((N,), dtype=np.float64)

    target = np.asarray(target_pts, dtype=np.float64)
    B_tree = cKDTree(target)

    shape_loss = np.empty((N,), dtype=np.float64)
    for i in range(N):
        shape_loss[i] = chamfer_distance_kdtree(pts0[i], B_tree, target)

    return -(shape_loss + w_close * closure + w_smooth * rough)


# -------------------------
# VAE architecture (must match your checkpoint)
# -------------------------
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
        self.norms  = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
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
            nn.Tanh(),   # outputs in [-1,1]
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
    return vae, L, latent_dim

@torch.no_grad()
def encode_dataset_to_z(vae, X_radians: np.ndarray, device, n_points: int, seed: int):
    N = X_radians.shape[0]
    rng = np.random.default_rng(seed)
    n = min(n_points, N)
    idx = rng.choice(N, size=n, replace=False)

    X_scaled = (X_radians[idx] / np.pi).astype(np.float32)
    Xt = torch.from_numpy(X_scaled).to(device)

    Z_chunks = []
    bs = 512
    for i in range(0, n, bs):
        mu, _ = vae.encode(Xt[i:i+bs])
        Z_chunks.append(mu.cpu().numpy())
    return np.vstack(Z_chunks)

@torch.no_grad()
def decode_batch_z_to_x_radians(vae, Z_real: np.ndarray, device) -> np.ndarray:
    Zt = torch.tensor(Z_real, dtype=torch.float32, device=device)
    x_scaled = vae.decode(Zt)                           # (N,L) in [-1,1]
    X_rad = x_scaled.detach().cpu().numpy().astype(np.float64) * np.pi
    return X_rad


# -------------------------
# Heatmap builder (run once)
# -------------------------
@torch.no_grad()
def oracle_heatmap_latent_grid(
    save_path: str,
    vae,
    device,
    target_pts: np.ndarray,
    step_size: float,
    w_close: float,
    w_smooth: float,
    z_box: float = 6.0,
    grid_res: int = 140,
    decode_bs: int = 4096,
    Z_data: np.ndarray | None = None,
    title: str = "True oracle heatmap: f(decode(z))"
):
    z1 = np.linspace(-z_box, z_box, grid_res)
    z2 = np.linspace(-z_box, z_box, grid_res)
    grid_x, grid_y = np.meshgrid(z1, z2)
    Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (G,2)

    # decode in batches
    X_chunks = []
    for i in range(0, Z_grid.shape[0], decode_bs):
        Zb = Z_grid[i:i+decode_bs]
        Xb = decode_batch_z_to_x_radians(vae, Zb, device)        # (bs,L) radians
        X_chunks.append(Xb)
    X_grid = np.vstack(X_chunks)                                 # (G,L)

    # true oracle
    y_grid = oracle_f_batch(X_grid, target_pts, step_size, w_close=w_close, w_smooth=w_smooth)
    Y = y_grid.reshape(grid_res, grid_res)

    # --- min (and max) on the grid + coordinates ---
    min_flat = int(np.argmin(Y))
    min_i, min_j = np.unravel_index(min_flat, Y.shape)  # i=row (z2), j=col (z1)
    y_min = float(Y[min_i, min_j])
    z_min = np.array([grid_x[min_i, min_j], grid_y[min_i, min_j]], dtype=float)

    max_flat = int(np.argmax(Y))
    max_i, max_j = np.unravel_index(max_flat, Y.shape)
    y_max = float(Y[max_i, max_j])
    z_max = np.array([grid_x[max_i, max_j], grid_y[max_i, max_j]], dtype=float)

    print(f"[GRID] min oracle = {y_min:.6f} at z = ({z_min[0]:.6f}, {z_min[1]:.6f})")
    print(f"[GRID] max oracle = {y_max:.6f} at z = ({z_max[0]:.6f}, {z_max[1]:.6f})")

    # plot
    fig = plt.figure(figsize=(6.6, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.contourf(grid_x, grid_y, Y, levels=45)
    plt.colorbar(im, ax=ax, label="oracle f(decode(z))")

    if Z_data is not None:
        ax.scatter(Z_data[:, 0], Z_data[:, 1], s=8, alpha=0.10, linewidth=0.0, label="data (encoded)")
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    
    ax.scatter([z_min[0]], [z_min[1]], s=120, marker="X", edgecolor="black", linewidth=1.2, label="grid min")
    ax.scatter([z_max[0]], [z_max[1]], s=120, marker="X", edgecolor="red", linewidth=1.2, label="grid max")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# -------------------------
# Main (run once)
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--z_box", type=float, default=6.0)
    ap.add_argument("--grid_res", type=int, default=140)
    ap.add_argument("--decode_bs", type=int, default=4096)

    ap.add_argument("--overlay_data", action="store_true", help="overlay encoded dataset cloud")
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generator first."
    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)          # (N,L) radians
    target = data["target"].astype(np.float64)

    # read oracle config (same as your other scripts)
    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        step_size = float(cfg.get("step_size", 1.0))
        w_close = float(cfg.get("w_close", 0.2))
        w_smooth = float(cfg.get("w_smooth", 0.05))
    else:
        step_size, w_close, w_smooth = 1.0, 0.2, 0.05

    vae_path = args.vae_path or os.path.join(outdir, "models", "vae.pt")
    assert os.path.exists(vae_path), f"Missing VAE at {vae_path}"
    vae, L, z_dim = load_vae(vae_path, device)
    assert z_dim == 2, "This script assumes latent_dim=2."
    assert X.shape[1] == L, f"Dataset L={X.shape[1]} but VAE expects L={L}"
    print(f"Loaded VAE: L={L}, z_dim={z_dim}")

    # output dir
    plot_root = os.path.join(outdir, "oracle_heatmap", f"z{args.z_box}_res{args.grid_res}")
    ensure_dir(plot_root)
    save_path = os.path.join(plot_root, "oracle_heatmap.png")

    Z_data = None
    if args.overlay_data:
        Z_data = encode_dataset_to_z(
            vae, X, device,
            n_points=args.n_data_scatter,
            seed=args.data_scatter_seed
        )

    print(f"Computing oracle grid on z ∈ [-{args.z_box},{args.z_box}]^2 with res={args.grid_res} ...")
    oracle_heatmap_latent_grid(
        save_path=save_path,
        vae=vae,
        device=device,
        target_pts=target,
        step_size=step_size,
        w_close=w_close,
        w_smooth=w_smooth,
        z_box=args.z_box,
        grid_res=args.grid_res,
        decode_bs=args.decode_bs,
        Z_data=Z_data,
        title="True oracle heatmap: f(decode(z))"
    )

    print("Saved:", os.path.abspath(save_path))


if __name__ == "__main__":
    main()
