# 04_cowboys_latent_diffprior.py
# COWBOYS-style BO with:
#   - Deterministic VAE decoder: x = h_theta(z)
#   - GP surrogate in SEQUENCE space x ∈ R^L (Matérn-5/2 ARD)
#   - Candidate generation via MCMC in latent space z
#   - BUT: the PRIOR over z is NOT N(0,I). It is a trained latent diffusion model p0(z).
#
# Key idea:
#   Target over z:  π(z) ∝ p0(z) * w_n(h(z))^tau
#     where w_n is PI or EI (default PI).
#
# We avoid needing log p0(z) by using an Independent Metropolis–Hastings (iMH) sampler:
#   propose z' ~ q(z') = p0(z')  (sample from latent diffusion)
#   accept with α = min(1, [w(h(z')) / w(h(z))]^tau )     (prior cancels because q=p0)
#
# Usage example:
#   python 08_latent_diffusion_cowboys.py --outdir toy_circle_data --n_steps 60 --acq pi --kappa 2.2 --weight pi --tau 1 --imh_steps 2500 --imh_burn 800 --imh_thin 10

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
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

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

def probability_of_improvement(mu, sigma, best_y, xi=0.05):
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - best_y - xi) / sigma
    return norm_cdf(Z)

def ucb(mu, sigma, kappa=2.0):
    return mu + kappa * sigma


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
            nn.Tanh(),  # outputs in [-1,1]
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
    train_types = ckpt.get("train_types", None)  # optional filter
    return vae, L, latent_dim, train_types


# =========================
# Encode/Decode helpers
# =========================
@torch.no_grad()
def decode_z_to_delta_theta(vae, z_np: np.ndarray, device) -> np.ndarray:
    z = torch.tensor(z_np, dtype=torch.float32, device=device).view(1, -1)
    x_hat_scaled = vae.decode(z)          # in [-1,1]
    return (x_hat_scaled.cpu().numpy().reshape(-1) * np.pi).astype(np.float64)

@torch.no_grad()
def encode_x_to_zmu(vae, x_radians: np.ndarray, device) -> np.ndarray:
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device).view(1, -1)
    mu, _ = vae.encode(xt)
    return mu.cpu().numpy().reshape(-1)

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


# =========================
# Latent diffusion prior p0(z) in VAE latent space
# =========================
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class EpsMLP(nn.Module):
    def __init__(self, z_dim: int, time_dim: int = 32, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.temb = SinusoidalTimeEmb(time_dim)
        layers = []
        d_in = z_dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        layers += [nn.Linear(hidden, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        te = self.temb(t)
        x = torch.cat([zt, te], dim=1)
        return self.net(x)

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)

def torch_load_compat(path: str, device):
    # PyTorch >=2.6 can default to weights_only=True, which breaks older checkpoints
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def load_latent_diffusion(diff_path: str, device):
    ckpt = torch_load_compat(diff_path, device)

    T = int(ckpt["T"])
    beta_start = float(ckpt["beta_start"])
    beta_end = float(ckpt["beta_end"])
    z_dim = int(ckpt.get("z_dim", 2))
    time_dim = int(ckpt.get("time_dim", 32))
    hidden = int(ckpt.get("hidden", 256))
    depth = int(ckpt.get("depth", 4))

    model = EpsMLP(z_dim=z_dim, time_dim=time_dim, hidden=hidden, depth=depth).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    z_mean = ckpt["z_mean"].astype(np.float32).reshape(1, -1)
    z_std  = ckpt["z_std"].astype(np.float32).reshape(1, -1)

    betas = make_beta_schedule(T, beta_start, beta_end).astype(np.float32)
    return model, betas, z_mean, z_std, T

@torch.no_grad()
def ddpm_arrays(betas: np.ndarray, device):
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = 1.0 - betas_t
    abar = torch.cumprod(alphas, dim=0)
    return betas_t, alphas, abar

@torch.no_grad()
def sample_z_from_latent_diffusion_prior(
    eps_model: nn.Module,
    betas: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device,
    z_dim: int = 2
) -> np.ndarray:
    """
    Unconditional DDPM sampling in *normalized* latent space, then unnormalize to VAE latent space.

    Returns:
      z_real: (z_dim,) numpy float64  in VAE latent coordinates.
    """
    betas_t, alphas, abar = ddpm_arrays(betas, device)
    T = betas_t.shape[0]

    z = torch.randn(1, z_dim, device=device)  # z_T (normalized coords)

    z_mean_t = torch.tensor(z_mean, device=device, dtype=torch.float32)  # (1,z_dim)
    z_std_t  = torch.tensor(z_std, device=device, dtype=torch.float32)   # (1,z_dim)

    for t in reversed(range(T)):
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        eps_hat = eps_model(z, tt)

        beta = betas_t[t]
        alpha = alphas[t]
        ab = abar[t]

        if t > 0:
            ab_prev = abar[t-1]
            beta_tilde = beta * (1.0 - ab_prev) / torch.clamp(1.0 - ab, min=1e-12)
            sigma = torch.sqrt(torch.clamp(beta_tilde, min=1e-20))
            noise = torch.randn_like(z)
        else:
            sigma = 0.0
            noise = torch.zeros_like(z)

        mean = (1.0 / torch.sqrt(alpha)) * (z - (beta / torch.sqrt(torch.clamp(1.0 - ab, min=1e-12))) * eps_hat)
        z = mean + sigma * noise

    z0_norm = z  # (1,z_dim)
    z_real = (z0_norm * z_std_t + z_mean_t).detach().cpu().numpy().reshape(-1).astype(np.float64)
    return z_real


# =========================
# Weight (w_n) used in COWBOYS "likelihood"
# =========================
def weight_value(
    gp: GaussianProcessRegressor,
    x: np.ndarray,
    best_y: float,
    weight: str,
    xi: float,
    kappa: float
) -> float:
    """
    w_n(x) used in target π(z) ∝ p0(z) * w_n(h(z))^tau
    Supported:
      - "pi":  PI(x)
      - "ei":  EI(x)
      - "ucb": exp-style target uses a(x)=UCB directly (handled elsewhere)
    """
    mu, std = gp.predict(x.reshape(1, -1), return_std=True)
    mu = float(mu[0]); std = float(std[0])

    if weight == "pi":
        return float(probability_of_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
    elif weight == "ei":
        return float(expected_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
    elif weight == "ucb":
        # treat as an acquisition "energy"; caller will use exp(tau * a(x))
        return float(ucb(mu, std, kappa=kappa))
    else:
        raise ValueError("weight must be one of: pi, ei, ucb")

def log_target_energy_from_weight(w: float, tau: float, weight: str, eps: float = 1e-12) -> float:
    """
    For MCMC acceptance we need log π up to constant.
      if weight in {pi, ei}: log π ∝ tau * log(w + eps)
      if weight == 'ucb':    log π ∝ tau * w  (since we want exp(tau * UCB))
    """
    if weight in ("pi", "ei"):
        return float(tau * np.log(max(w, eps)))
    elif weight == "ucb":
        return float(tau * w)
    else:
        raise ValueError("weight must be one of: pi, ei, ucb")


# =========================
# COWBOYS candidate generator:
# Independent MH in z with proposal z' ~ p0(z) (diffusion prior)
# Target: π(z) ∝ p0(z) * w(h(z))^tau
# Proposal: q(z') = p0(z')  => prior cancels; accept ratio uses only weights
# =========================
def imh_sample_candidates_diffusion_prior(
    gp: GaussianProcessRegressor,
    vae: nn.Module,
    eps_model: nn.Module,
    betas: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device,
    z0: np.ndarray,
    n_steps: int,
    burn: int,
    thin: int,
    tau: float,
    weight: str,
    xi: float,
    kappa: float,
    rng: np.random.Generator
):
    """
    Returns:
      samples: (M, d) latent samples (real z)
      chain:   (K, d) thinned chain for plotting
      w_chain: (K,)  weight values along chain
      accept_rate: float
    """
    d = z0.shape[0]
    z = z0.copy()

    # current weight/energy
    x = decode_z_to_delta_theta(vae, z, device)
    w_cur = weight_value(gp, x, best_y=float(gp.y_train_.max()), weight=weight, xi=xi, kappa=kappa)
    E_cur = log_target_energy_from_weight(w_cur, tau=tau, weight=weight)

    accepted = 0
    samples, chain, w_chain = [], [], []

    for t in range(1, n_steps + 1):
        # propose from diffusion prior (independent of current)
        z_prop = sample_z_from_latent_diffusion_prior(
            eps_model=eps_model, betas=betas, z_mean=z_mean, z_std=z_std,
            device=device, z_dim=d
        )

        x_prop = decode_z_to_delta_theta(vae, z_prop, device)
        w_prop = weight_value(gp, x_prop, best_y=float(gp.y_train_.max()), weight=weight, xi=xi, kappa=kappa)
        E_prop = log_target_energy_from_weight(w_prop, tau=tau, weight=weight)

        # since q(z') = p0(z') = prior, prior cancels in MH ratio:
        log_alpha = E_prop - E_cur

        if np.log(rng.random()) < min(0.0, log_alpha):
            z = z_prop
            w_cur = w_prop
            E_cur = E_prop
            accepted += 1

        if t % thin == 0:
            chain.append(z.copy())
            w_chain.append(w_cur)

        if t > burn and (t - burn) % thin == 0:
            samples.append(z.copy())

    samples = np.array(samples) if len(samples) else np.zeros((0, d))
    chain = np.array(chain) if len(chain) else np.zeros((0, d))
    w_chain = np.array(w_chain) if len(w_chain) else np.zeros((0,))
    accept_rate = accepted / max(1, n_steps)
    return samples, chain, w_chain, accept_rate


# =========================
# Visualizations
# =========================
def plot_step(save_path,
              Z_data, Z_obs, y_obs,
              z_next, y_next,
              z_best, y_best,
              chain, cand,
              grid_x, grid_y, A_grid, A_name,
              x_next, step_size,
              best_so_far,
              accept_rate):
    fig = plt.figure(figsize=(16, 5))

    # (1) Acquisition heatmap in latent coords: A(z) = A(x=h(z))
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(grid_x, grid_y, A_grid, levels=35)
    plt.colorbar(im, ax=ax1, label=A_name)

    ax1.scatter(Z_data[:,0], Z_data[:,1], s=8, alpha=0.12, linewidth=0.0, label="data (encoded)")
    ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55, edgecolor="black", linewidth=0.4, label="evaluated z")

    if chain.shape[0] > 0:
        ax1.plot(chain[:,0], chain[:,1], linewidth=1.2, alpha=0.9, label=f"iMH chain (acc={accept_rate:.2f})")
    if cand.shape[0] > 0:
        ax1.scatter(cand[:,0], cand[:,1], s=30, alpha=0.8, marker="x", label="iMH candidates")

    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label=f"best y={y_best:.2f}")
    ax1.scatter([z_next[0]], [z_next[1]], c="yellow", s=160, marker="D", edgecolor="black", linewidth=1.0,
                label=f"next y={y_next:.2f}")

    ax1.set_title("COWBOYS w/ diffusion prior: acquisition + iMH samples")
    ax1.set_xlabel("z1"); ax1.set_ylabel("z2")
    ax1.legend(loc="best")

    # (2) Decoded path for z_next
    ax2 = fig.add_subplot(1, 3, 2)
    pts = turtle_path(x_next, step_size)
    ax2.plot(pts[:,0], pts[:,1], linewidth=2.5)
    ax2.scatter([pts[0,0]], [pts[0,1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Decoded path at z_next\noracle f(x)={y_next:.3f}")

    # (3) Best-so-far curve
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(best_so_far, marker="o")
    ax3.set_title("Best-so-far oracle")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("best f(decode(z))")

    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close(fig)

def plot_final_summary(save_path, Z_data, Z_obs, y_obs, z_best, x_best, y_best, step_size):
    fig = plt.figure(figsize=(14, 4.8))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(Z_data[:,0], Z_data[:,1], s=10, alpha=0.15, linewidth=0.0, label="data (encoded)")
    sc = ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55, edgecolor="black", linewidth=0.4, label="BO points")
    plt.colorbar(sc, ax=ax1, label="oracle f(decode(z))")
    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label="best")
    ax1.set_title("Final latent overlay (data + BO + best)")
    ax1.set_xlabel("z1"); ax1.set_ylabel("z2")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 3, 2)
    pts = turtle_path(x_best, step_size)
    ax2.plot(pts[:,0], pts[:,1], linewidth=2.5)
    ax2.scatter([pts[0,0]], [pts[0,1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Best decoded path\noracle f(x)={y_best:.3f}")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(x_best)
    ax3.axhline(0.0, linestyle=":", linewidth=1)
    ax3.set_title("Best Δθ sequence")
    ax3.set_xlabel("t"); ax3.set_ylabel("Δθ (radians)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)
    ap.add_argument("--diff_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=60)

    # selection acquisition among sampled candidates (this is your BO choice)
    ap.add_argument("--acq", type=str, default="ucb", choices=["ucb", "ei", "pi"])
    ap.add_argument("--kappa", type=float, default=2.2)   # UCB
    ap.add_argument("--xi", type=float, default=0.06)     # EI / PI

    # COWBOYS "likelihood" weight for MCMC target
    ap.add_argument("--weight", type=str, default="pi", choices=["pi", "ei", "ucb"])
    ap.add_argument("--tau", type=float, default=40.0)    # sharpening/temperature for sampling target

    # iMH parameters (replacing pCN)
    ap.add_argument("--imh_steps", type=int, default=2500)
    ap.add_argument("--imh_burn", type=int, default=800)
    ap.add_argument("--imh_thin", type=int, default=10)

    # candidate usage (extra exploration outside MCMC samples)
    ap.add_argument("--n_random_cand", type=int, default=300,
                    help="extra candidates drawn directly from diffusion prior (no MH)")
    ap.add_argument("--eps_random_pick", type=float, default=0.10)

    # plotting grid (latent)
    ap.add_argument("--z_box", type=float, default=5.0)
    ap.add_argument("--grid_res", type=int, default=140)

    # background scatter
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    # init mode
    ap.add_argument("--init_mode", type=str, default="diffusion", choices=["data", "diffusion"],
                    help="data: seed from dataset top-y; diffusion: seed from diffusion prior")

    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generation first."

    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)        # (N,L) radians
    y_data = data["y"].astype(np.float64)   # (N,)
    target = data["target"].astype(np.float64)
    types = data["types"].astype(np.int64)

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
    diff_path = args.diff_path or os.path.join(outdir, "models", "latent_diffusion.pt")
    assert os.path.exists(vae_path), f"Missing VAE at {vae_path}. Train it first."
    assert os.path.exists(diff_path), f"Missing diffusion at {diff_path}. Train it first."

    vae, L, latent_dim, train_types = load_vae(vae_path, device)
    assert latent_dim == 2, "This script assumes latent_dim=2 for 2D visualizations."
    assert X.shape[1] == L, f"Dataset L={X.shape[1]} but VAE expects L={L}"
    print(f"Loaded VAE: L={L}, latent_dim={latent_dim}, train_types={train_types}")

    eps_model, betas, z_mean, z_std, T = load_latent_diffusion(diff_path, device)
    print(f"Loaded latent diffusion: T={T}, betas∈[{betas.min():.2e},{betas.max():.2e}], z_mean={z_mean}, z_std={z_std}")

    # output dirs
    plot_root = os.path.join(outdir, "plots_cowboys_diffprior")
    step_dir = os.path.join(plot_root, "steps")
    ensure_dir(step_dir)

    # background latent cloud (dataset encoded)
    Z_data = encode_dataset_to_z(
        vae, X, device,
        n_points=args.n_data_scatter,
        seed=args.data_scatter_seed
    )

    # init pool based on train_types if present (optional)
    if train_types is not None:
        allowed = sorted({int(c) for c in str(train_types)})
        mask = np.isin(types, allowed)
        pool_idx = np.where(mask)[0]
        print("Init pool types:", allowed, "pool size:", len(pool_idx))
    else:
        pool_idx = np.arange(len(X))

    # -------------------------
    # Initialize evaluated set
    # -------------------------
    Z_obs, y_obs = [], []

    if args.init_mode == "data":
        top_pool = pool_idx[np.argsort(y_data[pool_idx])[-max(args.n_init * 8, 200):]]
        init_idx = top_pool[np.argsort(y_data[top_pool])[-args.n_init:]]

        for idx in init_idx:
            x = X[idx]
            z_mu = encode_x_to_zmu(vae, x, device)
            x_dec = decode_z_to_delta_theta(vae, z_mu, device)
            y_dec = oracle_f(x_dec, target, step_size, w_close, w_smooth)
            Z_obs.append(z_mu); y_obs.append(y_dec)

    else:  # diffusion init
        for _ in range(args.n_init):
            z0 = sample_z_from_latent_diffusion_prior(eps_model, betas, z_mean, z_std, device, z_dim=2)
            x0 = decode_z_to_delta_theta(vae, z0, device)
            y0 = oracle_f(x0, target, step_size, w_close, w_smooth)
            Z_obs.append(z0); y_obs.append(y0)

    Z_obs = np.array(Z_obs, dtype=float)
    y_obs = np.array(y_obs, dtype=float)

    print("[INIT] best decoded y:", float(y_obs.max()))
    best_so_far = [float(y_obs.max())]

    # -------------------------
    # BO / COWBOYS iterations
    # -------------------------
    for it in range(1, args.n_steps + 1):

        # ---- Fit GP surrogate in SEQUENCE space x = h(z) ----
        # Build X_train by decoding each observed z
        X_train = np.stack([decode_z_to_delta_theta(vae, z, device) for z in Z_obs], axis=0)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(L), nu=2.5, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=args.seed
        )
        gp.fit(X_train, y_obs)

        best_y = float(y_obs.max())
        best_idx = int(np.argmax(y_obs))
        z_best = Z_obs[best_idx]
        y_best = float(y_obs[best_idx])

        # ---- Acquisition grid in latent space for plotting: A(z)=A(h(z)) ----
        z1 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        z2 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        grid_x, grid_y = np.meshgrid(z1, z2)
        Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # decode grid -> X grid
        X_grid = np.stack([decode_z_to_delta_theta(vae, z, device) for z in Z_grid], axis=0)
        mu_g, std_g = gp.predict(X_grid, return_std=True)

        if args.acq == "pi":
            A = probability_of_improvement(mu_g, std_g, best_y=best_y, xi=args.xi)
            A_name = "PI(x=h(z))"
        elif args.acq == "ei":
            A = expected_improvement(mu_g, std_g, best_y=best_y, xi=args.xi)
            A_name = "EI(x=h(z))"
        else:
            A = ucb(mu_g, std_g, kappa=args.kappa)
            A_name = "UCB(x=h(z))"
        A_grid = A.reshape(args.grid_res, args.grid_res)

        # ---- COWBOYS sampling step (MCMC) with diffusion prior proposal ----
        # chain starts at current best (standard in your toy)
        cand_mcmc, chain, w_chain, acc_rate = imh_sample_candidates_diffusion_prior(
            gp=gp,
            vae=vae,
            eps_model=eps_model,
            betas=betas,
            z_mean=z_mean,
            z_std=z_std,
            device=device,
            z0=z_best.copy(),
            n_steps=args.imh_steps,
            burn=args.imh_burn,
            thin=args.imh_thin,
            tau=args.tau,
            weight=args.weight,
            xi=args.xi,
            kappa=args.kappa,
            rng=rng
        )

        # extra exploration candidates drawn from diffusion prior directly
        cand_rand = np.stack([
            sample_z_from_latent_diffusion_prior(eps_model, betas, z_mean, z_std, device, z_dim=2)
            for _ in range(args.n_random_cand)
        ], axis=0).astype(np.float64)

        cand = np.vstack([cand_mcmc, cand_rand]) if cand_mcmc.shape[0] else cand_rand

        # ---- Pick next z among candidates by selection acquisition (args.acq) ----
        X_cand = np.stack([decode_z_to_delta_theta(vae, z, device) for z in cand], axis=0)
        mu_c, std_c = gp.predict(X_cand, return_std=True)

        if args.acq == "pi":
            A_c = probability_of_improvement(mu_c, std_c, best_y=best_y, xi=args.xi)
        elif args.acq == "ei":
            A_c = expected_improvement(mu_c, std_c, best_y=best_y, xi=args.xi)
        else:
            A_c = ucb(mu_c, std_c, kappa=args.kappa)

        if rng.random() < args.eps_random_pick:
            z_next = cand[rng.integers(0, len(cand))]
        else:
            z_next = cand[int(np.argmax(A_c))]

        # ---- Evaluate oracle at decode(z_next) ----
        x_next = decode_z_to_delta_theta(vae, z_next, device)
        y_next = float(oracle_f(x_next, target, step_size, w_close, w_smooth))

        # update BO dataset (in latent space)
        Z_obs = np.vstack([Z_obs, z_next.reshape(1, -1)])
        y_obs = np.append(y_obs, y_next)

        best_so_far.append(float(y_obs.max()))

        if it == 1 or it % 5 == 0:
            print(f"[COWBOYS+diff-prior iMH] step {it:3d} | y_next={y_next: .4f} | best={float(y_obs.max()): .4f} | acc={acc_rate:.2f}")

        # ---- Per-step plot ----
        save_path = os.path.join(step_dir, f"step_{it:03d}.png")
        best_idx = int(np.argmax(y_obs))
        z_best = Z_obs[best_idx]
        y_best = float(y_obs[best_idx])

        plot_step(
            save_path=save_path,
            Z_data=Z_data,
            Z_obs=Z_obs, y_obs=y_obs,
            z_next=z_next, y_next=y_next,
            z_best=z_best, y_best=y_best,
            chain=chain,
            cand=cand_mcmc[:400] if cand_mcmc.shape[0] else cand_mcmc,
            grid_x=grid_x, grid_y=grid_y, A_grid=A_grid, A_name=A_name,
            x_next=x_next, step_size=step_size,
            best_so_far=best_so_far,
            accept_rate=acc_rate
        )

    # -------------------------
    # Final summary
    # -------------------------
    best_idx = int(np.argmax(y_obs))
    z_best = Z_obs[best_idx]
    x_best = decode_z_to_delta_theta(vae, z_best, device)
    y_best = float(oracle_f(x_best, target, step_size, w_close, w_smooth))

    final_png = os.path.join(plot_root, "final_summary.png")
    plot_final_summary(final_png, Z_data, Z_obs, y_obs, z_best, x_best, y_best, step_size)

    trace_path = os.path.join(plot_root, "trace.npz")
    np.savez_compressed(trace_path, Z_obs=Z_obs, y_obs=y_obs, best_so_far=np.array(best_so_far))

    print("\n=== COWBOYS (diffusion prior) finished ===")
    print("Best oracle:", y_best)
    print("Best z:", z_best)
    print("Saved frames:", os.path.abspath(step_dir))
    print("Saved final summary:", os.path.abspath(final_png))
    print("Saved trace:", os.path.abspath(trace_path))


if __name__ == "__main__":
    main()
