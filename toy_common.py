"""Shared utilities for LSBO/COWBOYS scripts.

This module centralizes common math helpers, geometry/oracle functions,
and VAE encode/decode helpers to avoid copy-paste across experiment files.
"""

import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf
from scipy.spatial.distance import cdist


def set_seed(seed: int):
    """Set NumPy and PyTorch random seeds (CPU and CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def natural_key(s):
    """Natural-sort key, e.g. step_2 before step_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def norm_pdf(x):
    """Standard normal PDF."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def expected_improvement(mu, sigma, best_y, xi=0.05):
    """Expected Improvement acquisition for maximization."""
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best_y - xi
    z = imp / sigma
    return imp * norm_cdf(z) + sigma * norm_pdf(z)


def turtle_path(delta_theta: np.ndarray, step_size: float) -> np.ndarray:
    """Convert turn increments into a 2D polyline path."""
    delta_theta = np.asarray(delta_theta).reshape(-1)
    theta = 0.0
    p = np.array([0.0, 0.0], dtype=float)
    pts = [p.copy()]
    for dth in delta_theta:
        theta += float(dth)
        p = p + step_size * np.array([np.cos(theta), np.sin(theta)])
        pts.append(p.copy())
    return np.stack(pts, axis=0)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two 2D point clouds."""
    d = cdist(a, b)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def oracle_f(
    delta_theta: np.ndarray,
    target_pts: np.ndarray,
    step_size: float,
    w_close: float = 0.2,
    w_smooth: float = 0.05,
) -> float:
    """Oracle objective used in toy experiments (higher is better)."""
    pts = turtle_path(delta_theta, step_size=step_size)
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    shape_loss = chamfer_distance(pts0, target_pts)

    seq_len = len(delta_theta)
    closure = np.sum((pts[-1] - pts[0]) ** 2) / ((seq_len * step_size) ** 2)
    rough = np.mean(np.diff(delta_theta) ** 2) / (np.pi**2) if seq_len > 1 else 0.0
    return -(shape_loss + w_close * closure + w_smooth * rough)


class MLPBlock(nn.Module):
    """Linear + GELU + Dropout helper block."""

    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResMLP(nn.Module):
    """Residual MLP stack used by the VAE encoder."""

    def __init__(self, d_in, d_hidden, n_layers=4, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList(
            [MLPBlock(d_hidden, d_hidden, dropout) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_hidden)

    def forward(self, x):
        h = self.in_proj(x)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(ln(h))
        return self.out_norm(h)


class VAE(nn.Module):
    """VAE architecture expected by the trained checkpoints."""

    def __init__(
        self,
        L: int,
        latent_dim: int,
        hidden: int = 512,
        enc_layers: int = 6,
        dec_layers: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim

        self.encoder = ResMLP(L, hidden, n_layers=enc_layers, dropout=dropout)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        dec_blocks = []
        for _ in range(dec_layers):
            dec_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden),
                )
            )

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
    """Load VAE checkpoint and return model plus metadata."""
    ckpt = torch.load(vae_path, map_location=device)
    L = ckpt["L"]
    latent_dim = ckpt["latent_dim"]
    hidden = ckpt.get("hidden", 512)
    enc_layers = ckpt.get("enc_layers", 6)
    dec_layers = ckpt.get("dec_layers", 6)
    dropout = ckpt.get("dropout", 0.05)

    vae = VAE(
        L=L,
        latent_dim=latent_dim,
        hidden=hidden,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dropout=dropout,
    ).to(device)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.eval()

    train_types = ckpt.get("train_types", None)
    return vae, L, latent_dim, train_types


@torch.no_grad()
def decode_z_to_delta_theta(vae, z_np: np.ndarray, device) -> np.ndarray:
    """Decode latent z into angle increments (radians)."""
    z = torch.tensor(z_np, dtype=torch.float32, device=device).view(1, -1)
    x_hat_scaled = vae.decode(z)
    return (x_hat_scaled.cpu().numpy().reshape(-1) * np.pi).astype(np.float64)


@torch.no_grad()
def decode_z_to_x_radians(vae, z_np: np.ndarray, device) -> np.ndarray:
    """Alias for decode_z_to_delta_theta; kept for script compatibility."""
    return decode_z_to_delta_theta(vae, z_np, device)


@torch.no_grad()
def encode_x_to_zmu(vae, x_radians: np.ndarray, device) -> np.ndarray:
    """Encode a sequence and return latent posterior mean."""
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device).view(1, -1)
    mu, _ = vae.encode(xt)
    return mu.cpu().numpy().reshape(-1)


@torch.no_grad()
def encode_dataset_to_z(vae, x_radians: np.ndarray, device, n_points: int, seed: int):
    """Encode a random subset of sequences into latent means for plotting."""
    N = x_radians.shape[0]
    rng = np.random.default_rng(seed)
    n = min(n_points, N)
    idx = rng.choice(N, size=n, replace=False)

    x_scaled = (x_radians[idx] / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device)

    z_chunks = []
    bs = 512
    vae.eval()
    for i in range(0, n, bs):
        mu, _ = vae.encode(xt[i : i + bs])
        z_chunks.append(mu.cpu().numpy())
    return np.vstack(z_chunks)


def decode_deterministic(vae, z_np: np.ndarray, device) -> np.ndarray:
    """Backward-compatible decode helper name."""
    return decode_z_to_delta_theta(vae, z_np, device)


def encode_mu(vae, x_radians: np.ndarray, device) -> np.ndarray:
    """Backward-compatible encode helper name."""
    return encode_x_to_zmu(vae, x_radians, device)


def load_config(outdir: str):
    """Load step and oracle weights from config.json (with defaults)."""
    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return (
            float(cfg.get("step_size", 1.0)),
            float(cfg.get("w_close", 0.2)),
            float(cfg.get("w_smooth", 0.05)),
        )
    return 1.0, 0.2, 0.05
