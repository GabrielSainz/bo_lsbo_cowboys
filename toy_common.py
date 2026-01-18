import os, json
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist


# =========================
# Turtle + Oracle
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
    pts = turtle_path(delta_theta, step_size=step_size)
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    shape_loss = chamfer_distance(pts0, target_pts)

    L = len(delta_theta)
    closure = np.sum((pts[-1] - pts[0])**2) / ((L * step_size)**2)
    rough = np.mean(np.diff(delta_theta)**2) / (np.pi**2) if L > 1 else 0.0
    return -(shape_loss + w_close * closure + w_smooth * rough)


# =========================
# VAE (must match your trained checkpoint)
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

    train_types = ckpt.get("train_types", None)
    return vae, L, latent_dim, train_types

def decode_deterministic(vae, z_np: np.ndarray, device) -> np.ndarray:
    z = torch.tensor(z_np, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        x_hat_scaled = vae.decode(z)  # [-1,1]
    return (x_hat_scaled.cpu().numpy().reshape(-1) * np.pi).astype(np.float64)

def encode_mu(vae, x_radians: np.ndarray, device) -> np.ndarray:
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device).view(1, -1)
    with torch.no_grad():
        mu, _ = vae.encode(xt)
    return mu.cpu().numpy().reshape(-1)

def load_config(outdir: str):
    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("step_size", 1.0)), float(cfg.get("w_close", 0.2)), float(cfg.get("w_smooth", 0.05))
    return 1.0, 0.2, 0.05
