#python 03_lsbo.py --outdir toy_circle_data --n_steps 120 --xi 0.12 --eps_random 0.25 --z_box 6
#python 03_lsbo.py  --outdir toy_circle_data --n_steps 120 --xi 0.12 --eps_random 0.25 --z_box 3 --n_init 15 --n_data_scatter 3000


import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.special import erf
from scipy.spatial.distance import cdist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# =========================
# Turtle + oracle (must match dataset generator)
# =========================
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
    """
    Maximize this score.
    - shape loss: Chamfer distance of centered path to centered target
    - closure: end close to start
    - roughness: smoothness penalty on dtheta differences
    """
    pts = turtle_path(delta_theta, step_size=step_size)
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    shape_loss = chamfer_distance(pts0, target_pts)

    L = len(delta_theta)
    closure = np.sum((pts[-1] - pts[0])**2) / ((L * step_size)**2)

    rough = np.mean(np.diff(delta_theta)**2) / (np.pi**2) if L > 1 else 0.0
    return -(shape_loss + w_close * closure + w_smooth * rough)


# =========================
# VAE architecture (must match 02_train_vae.py checkpoint)
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

    train_types = ckpt.get("train_types", None)  # e.g. "01"
    return vae, L, latent_dim, train_types


# =========================
# Decode z -> Δθ (radians)
# =========================
def decode_z_to_delta_theta(vae, z_np: np.ndarray, device) -> np.ndarray:
    z = torch.tensor(z_np, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        x_hat_scaled = vae.decode(z)   # [-1,1]
    x_hat = (x_hat_scaled.cpu().numpy().reshape(-1) * np.pi).astype(np.float64)  # radians
    return x_hat

def encode_x_to_zmu(vae, x_radians: np.ndarray, device) -> np.ndarray:
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device).view(1, -1)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
    return mu.cpu().numpy().reshape(-1)


# =========================
# EI acquisition (maximization)
# =========================
def norm_pdf(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def expected_improvement(mu, sigma, best_y, xi=0.05):
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_y - xi
    Z = imp / sigma
    ei = imp * norm_cdf(Z) + sigma * norm_pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


# =========================
# Plotting
# =========================
def plot_step_2d(step_idx, save_path,Z_data,
                 grid_x, grid_y, EI_grid,
                 Z_obs, y_obs, z_next,
                 x_next, y_next,
                 best_so_far,
                 step_size):
    fig = plt.figure(figsize=(14, 4.8))

    # EI heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(grid_x, grid_y, EI_grid, levels=30)
    plt.colorbar(im, ax=ax1, label="EI")
    ax1.scatter(Z_data[:,0], Z_data[:,1], s=8, alpha=0.12, linewidth=0.0, label="data (encoded)")
    ax1.scatter(Z_obs[:,0], Z_obs[:,1], c="white", s=40, edgecolor="black", label="evaluated z")
    ax1.scatter([z_next[0]], [z_next[1]], c="red", s=90, marker="*", label="next z")
    ax1.set_title(f"Step {step_idx}: EI in latent space")
    ax1.set_xlabel("z1"); ax1.set_ylabel("z2")
    ax1.legend(loc="upper right")

    # decoded path
    ax2 = fig.add_subplot(1, 3, 2)
    pts = turtle_path(x_next, step_size)
    ax2.plot(pts[:,0], pts[:,1], linewidth=2)
    ax2.scatter([pts[0,0]], [pts[0,1]], s=60)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Decoded path @ z_next\noracle={y_next:.3f}")

    # sequence
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x_next)
    ax3.axhline(0.0, linestyle=":", linewidth=1)
    ax3.set_title("Sequence: Δθ_t")
    ax3.set_xlabel("t"); ax3.set_ylabel("Δθ (radians)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close(fig)

    # best-so-far curve
    plt.figure(figsize=(6,4))
    plt.plot(best_so_far, marker="o")
    plt.title("Best-so-far oracle")
    plt.xlabel("BO step")
    plt.ylabel("best f(decode(z))")
    plt.tight_layout()
    bpath = os.path.join(os.path.dirname(save_path), "..", "best_so_far.png")
    plt.savefig(os.path.abspath(bpath), dpi=170)
    plt.close()

def encode_dataset_to_z(vae, X_radians: np.ndarray, device, n_points: int, seed: int):
    """
    Encode a subset of dataset sequences to latent means for plotting.
    Returns Z_data (n_points, latent_dim).
    """
    N = X_radians.shape[0]
    rng = np.random.default_rng(seed)
    n = min(n_points, N)
    idx = rng.choice(N, size=n, replace=False)

    X_scaled = (X_radians[idx] / np.pi).astype(np.float32)
    Xt = torch.from_numpy(X_scaled).to(device)

    Z_chunks = []
    bs = 512
    vae.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            mu, _ = vae.encode(Xt[i:i+bs])
            Z_chunks.append(mu.cpu().numpy())
    Z_data = np.vstack(Z_chunks)
    return Z_data

def plot_final_summary(save_path,
                       Z_data,               # (Nd,2) latent means from dataset (subset)
                       Z_bo, y_bo,           # (Nb,2), (Nb,)
                       z_best,
                       x_best, y_best,
                       step_size):
    fig = plt.figure(figsize=(14, 4.8))

    # (A) Latent space: DATA + BO + BEST
    ax1 = fig.add_subplot(1, 3, 1)

    # DATA: light background cloud
    ax1.scatter(Z_data[:,0], Z_data[:,1],
                s=10, alpha=0.15, linewidth=0.0,
                label="data (encoded)")

    # BO points: colored by oracle
    sc = ax1.scatter(Z_bo[:,0], Z_bo[:,1],
                     c=y_bo, s=55, edgecolor="black", linewidth=0.4,
                     label="BO evaluated z")
    plt.colorbar(sc, ax=ax1, label="oracle f(decode(z)) (BO points)")

    # BEST: star
    ax1.scatter([z_best[0]], [z_best[1]],
                c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label="best BO z")

    ax1.set_title("Latent space overlay")
    ax1.set_xlabel("z1"); ax1.set_ylabel("z2")
    ax1.legend(loc="best")

    # (B) Best decoded path
    ax2 = fig.add_subplot(1, 3, 2)
    pts = turtle_path(x_best, step_size)
    ax2.plot(pts[:,0], pts[:,1], linewidth=2.5)
    ax2.scatter([pts[0,0]], [pts[0,1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Best decoded path\noracle f(x)={y_best:.3f}")

    # (C) Best Δθ sequence
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x_best)
    ax3.axhline(0.0, linestyle=":", linewidth=1)
    ax3.set_title("Best sequence: Δθ_t")
    ax3.set_xlabel("t"); ax3.set_ylabel("Δθ (radians)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

def save_generated_datapoint(gen_dir, step_idx, z_next, x_next, y_next, step_size):
    """
    Saves:
      - z_next (latent) as .npy
      - x_next (delta-theta sequence) as .npy
      - metadata as .json
      - decoded turtle path as .png
      - delta-theta plot as .png
    """
    ensure_dir(gen_dir)

    # ---- Save arrays
    np.save(os.path.join(gen_dir, f"z_{step_idx:03d}.npy"), np.asarray(z_next, dtype=float))
    np.save(os.path.join(gen_dir, f"x_{step_idx:03d}.npy"), np.asarray(x_next, dtype=float))

    meta = {
        "step": int(step_idx),
        "y_next": float(y_next),
        "z_next": np.asarray(z_next, dtype=float).tolist(),
    }
    with open(os.path.join(gen_dir, f"meta_{step_idx:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---- Plot turtle path
    pts = turtle_path(x_next, step_size=step_size)
    plt.figure(figsize=(5, 5))
    plt.plot(pts[:, 0], pts[:, 1], linewidth=2.5)
    plt.scatter([pts[0, 0]], [pts[0, 1]], s=70)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(f"step={step_idx} | y={y_next:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(gen_dir, f"path_{step_idx:03d}.png"), dpi=180)
    plt.close()

    # ---- Plot delta-theta sequence
    plt.figure(figsize=(7, 3))
    plt.plot(x_next)
    plt.axhline(0.0, linestyle=":", linewidth=1)
    plt.title(f"Δθ sequence | step={step_idx} | y={y_next:.3f}")
    plt.xlabel("t")
    plt.ylabel("Δθ (radians)")
    plt.tight_layout()
    plt.savefig(os.path.join(gen_dir, f"seq_{step_idx:03d}.png"), dpi=180)
    plt.close()


# =========================
# Main LSBO
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=60)

    ap.add_argument("--xi", type=float, default=0.08)
    ap.add_argument("--eps_random", type=float, default=0.20)   # exploration probability
    ap.add_argument("--topk_ei", type=int, default=30)

    ap.add_argument("--z_box", type=float, default=5.0)
    ap.add_argument("--grid_res", type=int, default=140)
    
    ap.add_argument("--n_data_scatter", type=int, default=3000,
                help="How many dataset points to encode+plot in latent space for final_summary.")
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    ap.add_argument("--cand_highd", type=int, default=4000)     # if latent_dim != 2
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generation first."

    # Load dataset
    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)        # (N,L) radians
    y_data = data["y"].astype(np.float64)   # oracle at X
    target = data["target"].astype(np.float64)
    types = data["types"].astype(np.int64)

    # Load config params if available
    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        step_size = float(cfg.get("step_size", 1.0))
        w_close = float(cfg.get("w_close", 0.2))
        w_smooth = float(cfg.get("w_smooth", 0.05))
    else:
        step_size, w_close, w_smooth = 1.0, 0.2, 0.05

    # Load VAE
    vae_path = args.vae_path or os.path.join(outdir, "models", "vae.pt")
    assert os.path.exists(vae_path), f"Missing VAE at {vae_path}. Train it first."
    vae, L, latent_dim, train_types = load_vae(vae_path, device)
    print(f"Loaded VAE: L={L}, latent_dim={latent_dim}, train_types={train_types}")

    assert X.shape[1] == L, f"Dataset L={X.shape[1]} but VAE expects L={L}"

    # Output dirs
    plot_root = os.path.join(outdir, "plots_lsbo2")
    step_dir = os.path.join(plot_root, "bo_steps")
    ensure_dir(plot_root)
    ensure_dir(step_dir)
    
    gen_dir = os.path.join(plot_root, "generated_datapoints")
    ensure_dir(gen_dir)

    Z_data = encode_dataset_to_z(
            vae, X, device,
            n_points=args.n_data_scatter,
            seed=args.data_scatter_seed
        )

    # Choose init pool: use the VAE training types if present, else use all
    if train_types is not None:
        allowed = sorted({int(c) for c in str(train_types)})
        mask = np.isin(types, allowed)
        pool_idx = np.where(mask)[0]
        print("Init pool types:", allowed, "pool size:", len(pool_idx))
    else:
        pool_idx = np.arange(len(X))

    # ----- Initialize Z_obs from top oracle examples in pool
    # (but evaluate y via decode(mu(x)) to stay consistent)
    top_pool = pool_idx[np.argsort(y_data[pool_idx])[-max(args.n_init * 8, 200):]]  # get a decent top set
    init_idx = top_pool[np.argsort(y_data[top_pool])[-args.n_init:]]

    Z_obs, y_obs, x_obs = [], [], []
    #for idx in init_idx:
    #    x = X[idx]
    #    z_mu = encode_x_to_zmu(vae, x, device)
    #    x_dec = decode_z_to_delta_theta(vae, z_mu, device)
    #    y_dec = oracle_f(x_dec, target, step_size, w_close, w_smooth)
    #
    #    Z_obs.append(z_mu)
    #    x_obs.append(x_dec)
    #    y_obs.append(y_dec)

    # Add a few random z init points too (helps exploration)
    #n_rand_init = max(3, args.n_init // 5)
    n_rand_init = args.n_init
    for _ in range(n_rand_init):
        z0 = np.random.randn(latent_dim)
        x0 = decode_z_to_delta_theta(vae, z0, device)
        y0 = oracle_f(x0, target, step_size, w_close, w_smooth)
        Z_obs.append(z0); x_obs.append(x0); y_obs.append(y0)

    Z_obs = np.array(Z_obs, dtype=float)
    y_obs = np.array(y_obs, dtype=float)

    print("\n[INIT] best decoded y (reachable):", float(y_obs.max()))
    best_so_far = [float(y_obs.max())]

    # ----- BO loop
    for t in range(1, args.n_steps + 1):

        # GP kernel (Matérn tends to behave better than pure RBF here)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(latent_dim), nu=2.5, length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=args.seed
        )
        gp.fit(Z_obs, y_obs)

        best_y = float(y_obs.max())

        # ---- Propose z_next
        if latent_dim == 2:
            # Grid EI for visualization
            z1 = np.linspace(-args.z_box, args.z_box, args.grid_res)
            z2 = np.linspace(-args.z_box, args.z_box, args.grid_res)
            grid_x, grid_y = np.meshgrid(z1, z2)
            Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

            mu, std = gp.predict(Z_grid, return_std=True)
            ei = expected_improvement(mu, std, best_y=best_y, xi=args.xi)
            EI_grid = ei.reshape(args.grid_res, args.grid_res)

            # epsilon-greedy + top-k EI sampling
            if np.random.rand() < args.eps_random or np.std(ei) < 1e-12:
                z_next = np.random.uniform(-args.z_box, args.z_box, size=(2,))
            else:
                k = min(args.topk_ei, len(ei))
                topk = np.argpartition(ei, -k)[-k:]
                z_next = Z_grid[np.random.choice(topk)]

        else:
            # High-dim: sample candidates and pick best EI
            if np.random.rand() < args.eps_random:
                z_next = np.random.randn(latent_dim)
            else:
                # mixture: normal + uniform
                Zcand = np.random.randn(args.cand_highd, latent_dim)
                Zcand2 = np.random.uniform(-args.z_box, args.z_box, size=(args.cand_highd // 2, latent_dim))
                Zcand = np.vstack([Zcand, Zcand2])

                mu, std = gp.predict(Zcand, return_std=True)
                ei = expected_improvement(mu, std, best_y=best_y, xi=args.xi)
                z_next = Zcand[int(np.argmax(ei))]

            grid_x = grid_y = EI_grid = None  # no 2D plot

        # ---- Evaluate oracle at decode(z_next)
        x_next = decode_z_to_delta_theta(vae, z_next, device)
        y_next = oracle_f(x_next, target, step_size, w_close, w_smooth)

        save_generated_datapoint(
            gen_dir=gen_dir,
            step_idx=t,
            z_next=z_next,
            x_next=x_next,
            y_next=y_next,
            step_size=step_size
        )

        # ---- Update
        Z_obs = np.vstack([Z_obs, z_next.reshape(1, -1)])
        y_obs = np.append(y_obs, float(y_next))
        x_obs.append(x_next)

        best_so_far.append(float(y_obs.max()))

        if t == 1 or t % 5 == 0:
            print(f"[BO] step {t:3d} | y_next={y_next: .4f} | best={float(y_obs.max()): .4f}")

        # Save plots
        if latent_dim == 2:
            save_path = os.path.join(step_dir, f"step_{t:03d}.png")
            plot_step_2d(
                step_idx=t,
                save_path=save_path,
                Z_data=Z_data,
                grid_x=grid_x, grid_y=grid_y, EI_grid=EI_grid,
                Z_obs=Z_obs, y_obs=y_obs, z_next=z_next,
                x_next=x_next, y_next=y_next,
                best_so_far=best_so_far,
                step_size=step_size
            )

    # ----- Save trace
    trace_path = os.path.join(plot_root, "bo_trace.npz")
    np.savez_compressed(
        trace_path,
        Z_obs=Z_obs,
        y_obs=y_obs,
        best_so_far=np.array(best_so_far, dtype=float),
    )

    best_idx = int(np.argmax(y_obs))
    z_best = Z_obs[best_idx]
    x_best = decode_z_to_delta_theta(vae, z_best, device)
    y_best = float(oracle_f(x_best, target, step_size, w_close, w_smooth))

    # Final summary plot (only if 2D latent)
    if latent_dim == 2:
        Z_data = encode_dataset_to_z(
            vae, X, device,
            n_points=args.n_data_scatter,
            seed=args.data_scatter_seed
        )
        final_png = os.path.join(plot_root, "final_summary.png")
        plot_final_summary(
            save_path=final_png,
            Z_data=Z_data,
            Z_bo=Z_obs, y_bo=y_obs,
            z_best=z_best,
            x_best=x_best, y_best=y_best,
            step_size=step_size
        )

    print("Saved final summary:", os.path.abspath(final_png))
    print("\n=== LSBO finished ===")
    print("Best oracle:", y_best)
    print("Best z:", z_best)

    # Save best path alone (optional)
    best_path_png = os.path.join(plot_root, "best_path.png")
    plt.figure(figsize=(5,5))
    pts = turtle_path(x_best, step_size)
    plt.plot(pts[:,0], pts[:,1], linewidth=2.5)
    plt.scatter([pts[0,0]], [pts[0,1]], s=70)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(f"Best path\noracle f(x)={y_best:.3f}")
    plt.tight_layout()
    plt.savefig(best_path_png, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
