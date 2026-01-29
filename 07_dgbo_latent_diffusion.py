# 07_dgbo_latent_diffusion.py
# Diffusion-Guided Bayesian Optimisation (DGBO) in latent space
# - Diffusion prior in latent space (DDPM) already trained (latent_diffusion.pt)
# - VAE already trained (vae.pt) used as deterministic decoder x = h_theta(z)
# - GP surrogate is fit in SEQUENCE space x ∈ R^L with Matérn-5/2 ARD kernel (same as your COWBOYS kernel)
# - Guidance uses ∇_{z_t} log w_n(h_theta(z0_hat)) where w_n is PI or EI (default PI)
#
# Usage example:
# python 07_dgbo_latent_diffusion.py --outdir toy_circle_data --n_steps 60 --n_init 15 --weight pi --xi 0.06 --guidance_scale 2.0 --guide_every 1 --n_cand 24 --tau_guidance 3
#
# Notes:
# - This script implements analytic GP gradients for Matérn-5/2 ARD and uses sklearn only to fit kernel hyperparams.
# - Guidance is applied through eps-guidance: eps_guided = eps - s * sqrt(1-abar_t) * grad_{z_t} log w
# - We approximate ∂z0_hat/∂z_t ≈ 1/sqrt(abar_t) I (ignoring eps-network Jacobian) for stability + speed.

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

# -------------------------
# Utils
# -------------------------
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
    return vae, L, latent_dim

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
    vae.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            mu, _ = vae.encode(Xt[i:i+bs])
            Z_chunks.append(mu.cpu().numpy())
    return np.vstack(Z_chunks)

@torch.no_grad()
def decode_z_to_x_radians(vae, z_real_np: np.ndarray, device) -> np.ndarray:
    z = torch.tensor(z_real_np, dtype=torch.float32, device=device).view(1, -1)
    x_scaled = vae.decode(z)              # [-1,1]
    x_rad = x_scaled * np.pi              # [-pi,pi]
    return x_rad.detach().cpu().numpy().reshape(-1).astype(np.float64)

# -------------------------
# Latent diffusion model (must match your training)
# -------------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device).float() / (half - 1)
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
    """
    PyTorch >=2.6 may default to weights_only=True, which breaks older checkpoints
    containing numpy arrays / other pickled objects.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # older torch versions don't have weights_only arg
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
    print("z_mean:", z_mean, z_mean.shape)
    print("z_std :", z_std, z_std.shape)
    print("T:", T, "betas:", betas.shape, betas.min(), betas.max())

    return model, betas, z_mean, z_std, T

# -------------------------
# GP + Matérn-5/2 ARD kernel gradients (sequence space)
# -------------------------
def parse_kernel_params(gp):
    """
    Expect gp.kernel_ ≈ ConstantKernel * Matern(nu=2.5) + WhiteKernel
    Return sigma_f2, length_scales (L,), sigma_n2.
    """
    k = gp.kernel_
    # Sum
    try:
        k1, k2 = k.k1, k.k2
    except Exception:
        raise RuntimeError(f"Unexpected kernel structure: {k}")

    # identify white kernel
    if isinstance(k1, WhiteKernel):
        white = k1
        prod = k2
    elif isinstance(k2, WhiteKernel):
        white = k2
        prod = k1
    else:
        raise RuntimeError("Kernel must include WhiteKernel as one summand.")

    if not hasattr(prod, "k1") or not hasattr(prod, "k2"):
        raise RuntimeError("Expected product kernel ConstantKernel * Matern.")
    if isinstance(prod.k1, ConstantKernel) and isinstance(prod.k2, Matern):
        ck, mk = prod.k1, prod.k2
    elif isinstance(prod.k2, ConstantKernel) and isinstance(prod.k1, Matern):
        ck, mk = prod.k2, prod.k1
    else:
        raise RuntimeError("Expected ConstantKernel and Matern in product.")

    sigma_f2 = float(ck.constant_value)
    ell = np.array(mk.length_scale, dtype=np.float64).reshape(-1)
    sigma_n2 = float(white.noise_level)

    if float(mk.nu) != 2.5:
        raise RuntimeError(f"Expected Matern nu=2.5 but got nu={mk.nu}")
    return sigma_f2, ell, sigma_n2

def matern52_ard_k_and_grad(x, X_train, sigma_f2, ell):
    """
    x: (L,) float64
    X_train: (n,L) float64
    Return:
      kvec: (n,)
      grad_k: (n,L) where grad_k[i,j] = ∂/∂x_j k(x, X_i)
    Matérn-5/2 ARD:
      r = sqrt(sum_j ((x_j-x'_j)^2 / ell_j^2))
      k = sigma_f2 (1 + a r + b r^2) exp(-a r), a=sqrt(5), b=5/3
      grad: ∂k/∂x_j = -sigma_f2*(5/3)*(1 + a r)*exp(-a r)*(x_j-x'_j)/ell_j^2   (for r>0)
    """
    x = x.reshape(1, -1)  # (1,L)
    diff = x - X_train    # (n,L)
    ell2 = (ell**2).reshape(1, -1)
    dscaled = diff / ell.reshape(1, -1)
    r = np.sqrt(np.sum(dscaled**2, axis=1))  # (n,)

    a = np.sqrt(5.0)
    b = 5.0/3.0

    # k
    poly = (1.0 + a*r + b*(r**2))
    exp_term = np.exp(-a*r)
    kvec = sigma_f2 * poly * exp_term

    # grad
    grad_k = np.zeros_like(diff, dtype=np.float64)  # (n,L)
    # handle r>0
    m = r > 1e-12
    if np.any(m):
        rm = r[m]
        factor = -sigma_f2 * (5.0/3.0) * (1.0 + a*rm) * np.exp(-a*rm)  # (m,)
        # multiply by (x-x')/ell^2
        grad_k[m, :] = (factor.reshape(-1,1)) * (diff[m, :] / ell2)
    return kvec.astype(np.float64), grad_k.astype(np.float64)

def gp_mu_sigma_and_grads(gp, x):
    """
    Compute GP posterior mean, std, and gradients wrt x (sequence-space).
    Uses gp.L_ (Cholesky of training K) and gp.alpha_ = K^{-1} y_train (in normalized y space if normalize_y=True).
    Returns:
      mu, sigma, grad_mu, grad_sigma
    """
    Xtr = gp.X_train_
    y_alpha = gp.alpha_.reshape(-1)  # (n,)

    # normalization factors (sklearn private attrs)
    y_mean = getattr(gp, "_y_train_mean", 0.0)
    y_std  = getattr(gp, "_y_train_std", 1.0)
    if np.isscalar(y_mean): y_mean = float(y_mean)
    if np.isscalar(y_std):  y_std = float(y_std)

    sigma_f2, ell, _sigma_n2 = parse_kernel_params(gp)

    # k(x, X_train) and grad
    kvec, grad_k = matern52_ard_k_and_grad(np.asarray(x, dtype=np.float64), Xtr, sigma_f2, ell)  # (n,), (n,L)

    # solve v = L^{-1} k
    # gp.L_ is lower-triangular (n,n)
    Lchol = gp.L_
    v = np.linalg.solve(Lchol, kvec)             # (n,)
    w = np.linalg.solve(Lchol.T, v)              # (n,) = K^{-1} k

    # posterior in normalized-y space
    mu_norm = float(kvec @ y_alpha)
    var_norm = float(max(1e-12, sigma_f2 - (v @ v)))  # k(x,x)=sigma_f2 for stationary signal kernel
    sigma_norm = float(np.sqrt(var_norm))

    # gradients in normalized-y space
    grad_mu_norm = grad_k.T @ y_alpha            # (L,)
    grad_var_norm = -2.0 * (grad_k.T @ w)        # (L,)
    grad_sigma_norm = 0.5 * grad_var_norm / max(sigma_norm, 1e-12)

    # unnormalize
    mu = mu_norm * y_std + y_mean
    sigma = sigma_norm * y_std
    grad_mu = grad_mu_norm * y_std
    grad_sigma = grad_sigma_norm * y_std

    return float(mu), float(max(sigma, 1e-12)), grad_mu.astype(np.float64), grad_sigma.astype(np.float64)

def grad_log_weight_PI(gp, x, best_y, xi=0.06, eps=1e-12):
    """
    w(x) = PI(x) = Phi( (mu - best_y - xi)/sigma )
    Return grad_x log w(x)
    """
    mu, sigma, grad_mu, grad_sigma = gp_mu_sigma_and_grads(gp, x)
    sigma = max(sigma, 1e-10)
    u = (mu - best_y - xi) / sigma

    Phi = norm_cdf(u)
    phi = norm_pdf(u)

    # grad u = (sigma*grad_mu - (mu-best_y-xi)*grad_sigma) / sigma^2
    num = sigma * grad_mu - (mu - best_y - xi) * grad_sigma
    grad_u = num / (sigma**2)

    # grad log Phi(u) = (phi/Phi) * grad_u
    return (phi / max(Phi, eps)) * grad_u

def expected_improvement(mu, sigma, best_y, xi=0.05):
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_y - xi
    Z = imp / sigma
    ei = imp * norm_cdf(Z) + sigma * norm_pdf(Z)
    ei = np.where(sigma < 1e-9, 0.0, ei)
    return ei

def grad_log_weight_EI(gp, x, best_y, xi=0.06, eps=1e-12):
    """
    w(x) = EI(x) = E[max(0, mu+sigma*Z - best_y - xi)]
    We guide with grad log(EI+eps).
    """
    mu, sigma, grad_mu, grad_sigma = gp_mu_sigma_and_grads(gp, x)
    sigma = max(sigma, 1e-10)

    imp = mu - best_y - xi
    u = imp / sigma

    Phi = norm_cdf(u)
    phi = norm_pdf(u)

    EI = imp * Phi + sigma * phi
    EI = max(float(EI), 0.0)

    # d(EI)/dmu = Phi(u)
    dEI_dmu = Phi
    # d(EI)/dsigma = phi(u)
    dEI_dsigma = phi

    grad_EI = dEI_dmu * grad_mu + dEI_dsigma * grad_sigma
    return grad_EI / max(EI, eps)

def gp_acquisition(gp, x, best_y, acq="ei", xi=0.06):
    mu, sigma, _gm, _gs = gp_mu_sigma_and_grads(gp, x)
    if acq == "mu":
        return mu
    elif acq == "pi":
        u = (mu - best_y - xi) / max(sigma, 1e-10)
        return float(norm_cdf(u))
    elif acq == "ei":
        return float(expected_improvement(np.array([mu]), np.array([sigma]), best_y=best_y, xi=xi)[0])
    else:
        raise ValueError("acq must be one of: mu, pi, ei")

# -------------------------
# Guided DDPM sampler in latent space
# -------------------------
@torch.no_grad()
def ddpm_arrays(betas: np.ndarray, device):
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = 1.0 - betas_t
    abar = torch.cumprod(alphas, dim=0)
    return betas_t, alphas, abar

def guided_sample_latent_z0_norm(
    eps_model: nn.Module,
    vae: nn.Module,
    gp: GaussianProcessRegressor,
    betas: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    best_y: float,
    xi: float,
    weight: str,
    guidance_scale: float,
    guide_every: int,
    clip_guidance: float,
    device,
    tau_guidance: float = 1.0,
    keep_traj: bool = False
):
    """
    Returns:
      z0_norm: (2,) numpy
      traj (optional): list of z_t (numpy) for plotting
    """
    betas_t, alphas, abar = ddpm_arrays(betas, device)
    T = betas_t.shape[0]

    z = torch.randn(1, 2, device=device)  # z_T in normalized latent space
    traj = [z.detach().cpu().numpy().reshape(-1)] if keep_traj else None

    # constants for guidance mapping
    z_mean_t = torch.tensor(z_mean, device=device, dtype=torch.float32)  # (1,2)
    z_std_t  = torch.tensor(z_std, device=device, dtype=torch.float32)   # (1,2)

    for t in reversed(range(T)):
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_hat = eps_model(z, tt)  # (1,2)

        eps_use = eps_hat

        # guidance (every k steps)
        if guidance_scale > 0.0 and (t % max(1, guide_every) == 0):
            ab = abar[t]
            sqrt_ab = torch.sqrt(ab)
            sqrt_1mab = torch.sqrt(1.0 - ab)

            # z0_hat in normalized coords (detach; we will ignore d eps_hat / d z_t)
            z0_hat_norm = (z - sqrt_1mab * eps_hat) / torch.clamp(sqrt_ab, min=1e-12)

            # unnormalize to VAE latent space
            z0_hat_real = (z0_hat_norm.detach() * z_std_t + z_mean_t).clone().detach().requires_grad_(True)

            # deterministic decode to x (radians)
            x_scaled = vae.decode(z0_hat_real)     # (1,L) in [-1,1]
            x_rad = x_scaled * np.pi               # (1,L)

            x_np = x_rad.detach().cpu().numpy().reshape(-1).astype(np.float64)

            # compute grad_x log w(x)
            if weight == "pi":
                grad_x = grad_log_weight_PI(gp, x_np, best_y=best_y, xi=xi)
            elif weight == "ei":
                grad_x = grad_log_weight_EI(gp, x_np, best_y=best_y, xi=xi)
            else:
                raise ValueError("weight must be 'pi' or 'ei'")
            
            grad_x = tau_guidance * grad_x
            grad_x_torch = torch.tensor(grad_x, device=device, dtype=torch.float32).view(1, -1)

            # vjp: J^T grad_x via scalar product
            scalar = (x_rad * grad_x_torch).sum()
            scalar.backward()
            grad_z0_real = z0_hat_real.grad.detach()  # (1,2)


            # map to grad w.r.t z_t_norm using ∂z0_real/∂z_t_norm ≈ diag(z_std)/sqrt(abar_t)
            grad_z_t = (z_std_t / torch.clamp(sqrt_ab, min=1e-12)) * grad_z0_real  # (1,2)

            # clip guidance (optional)
            if clip_guidance > 0.0:
                gnorm = torch.norm(grad_z_t, dim=1, keepdim=True).clamp(min=1e-12)
                grad_z_t = grad_z_t * torch.clamp(clip_guidance / gnorm, max=1.0)

            # eps-guidance: eps_guided = eps - s * sqrt(1-abar_t) * grad
            eps_use = eps_hat - guidance_scale * sqrt_1mab * grad_z_t

        # DDPM posterior sampling (correct beta_tilde)
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

        mean = (1.0 / torch.sqrt(alpha)) * (z - (beta / torch.sqrt(torch.clamp(1.0 - ab, min=1e-12))) * eps_use)
        z = mean + sigma * noise

        if keep_traj and (t in {T-1, int(0.8*(T-1)), int(0.6*(T-1)), int(0.4*(T-1)), int(0.2*(T-1)), 0}):
            traj.append(z.detach().cpu().numpy().reshape(-1))

    z0_norm = z.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return z0_norm, traj


@torch.no_grad()
def decode_batch_z_to_x_radians(vae, Z_real: np.ndarray, device) -> np.ndarray:
    """
    Z_real: (N,2) in VAE latent space
    returns X_radians: (N,L)
    """
    Zt = torch.tensor(Z_real, dtype=torch.float32, device=device)
    x_scaled = vae.decode(Zt)                 # (N,L) in [-1,1]
    X_rad = x_scaled.detach().cpu().numpy().astype(np.float64) * np.pi
    return X_rad


# -------------------------
# Plots
# -------------------------
def plot_step(save_path,
              Z_data, Z_obs, y_obs,
              Z_cand, z_next, y_next,
              z_best, y_best,
              x_next, step_size,
              best_so_far,
              grid_x=None, grid_y=None, A_grid=None, A_name=None,
              traj=None):

    fig = plt.figure(figsize=(16, 5))

    # (1) Acquisition heatmap + overlays
    ax1 = fig.add_subplot(1, 3, 1)
    if A_grid is not None:
        im = ax1.contourf(grid_x, grid_y, A_grid, levels=35)
        plt.colorbar(im, ax=ax1, label=A_name)
    else:
        # fallback: color by oracle
        sc = ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55,
                         edgecolor="black", linewidth=0.4)
        plt.colorbar(sc, ax=ax1, label="oracle y")

    ax1.scatter(Z_data[:,0], Z_data[:,1], s=8, alpha=0.12, linewidth=0.0, label="data (encoded)")
    ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55, edgecolor="black", linewidth=0.4, label="evaluated z")

    if Z_cand is not None and Z_cand.shape[0] > 0:
        ax1.scatter(Z_cand[:,0], Z_cand[:,1], s=30, alpha=0.8, marker="x",
                    label="guided diffusion samples (batch)")

    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label=f"best y={y_best:.3f}")
    ax1.scatter([z_next[0]], [z_next[1]], c="yellow", s=160, marker="D", edgecolor="black", linewidth=1.0,
                label=f"next y={y_next:.3f}")

    if traj is not None and len(traj) >= 2:
        tr = np.stack(traj, axis=0)
        ax1.plot(tr[:,0], tr[:,1], linewidth=1.2, alpha=0.9, label="one guided reverse traj")

    ax1.set_title("DGBO: acquisition + guided diffusion candidates")
    ax1.set_xlabel("z1"); ax1.set_ylabel("z2")
    ax1.legend(loc="best")

    # (2) Decoded path at z_next (THIS fixes your empty panel)
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
    sc = ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55, edgecolor="black", linewidth=0.4, label="DGBO points")
    plt.colorbar(sc, ax=ax1, label="oracle y")
    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label="best")
    ax1.set_title("Final latent overlay (data + DGBO + best)")
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

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)
    ap.add_argument("--diff_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    # DGBO loop
    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=60)

    # surrogate + weights
    ap.add_argument("--weight", type=str, default="pi", choices=["pi", "ei"])
    ap.add_argument("--xi", type=float, default=0.06)
    ap.add_argument("--select_acq", type=str, default="mu", choices=["mu", "pi", "ei"])

    # guided diffusion sampling
    ap.add_argument("--n_cand", type=int, default=24, help="batch of guided diffusion samples per BO iteration")
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--guide_every", type=int, default=5)
    ap.add_argument("--clip_guidance", type=float, default=5.0, help="max norm of guidance grad in z_t (0 disables)")

    # init mode
    ap.add_argument("--init_mode", type=str, default="data", choices=["data", "vae_prior", "diffusion"],
                    help="how to create initial evaluated set")

    # GP fitting
    ap.add_argument("--n_restarts", type=int, default=3)
    ap.add_argument("--max_train_gp", type=int, default=200, help="cap GP training set size (keeps kernel inversion stable)")

    # plotting
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    ap.add_argument("--z_box", type=float, default=3)
    ap.add_argument("--grid_res", type=int, default=140)
    ap.add_argument("--plot_acq", type=str, default="pi", choices=["mu","pi","ei"])

    ap.add_argument("--tau_guidance", type=float, default=1.0)

    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generator first."
    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)        # (N,L) radians
    y_data = data["y"].astype(np.float64)   # (N,)
    target = data["target"].astype(np.float64)
    types = data["types"].astype(np.int64) if "types" in data else None

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
    assert os.path.exists(vae_path), f"Missing VAE at {vae_path}"
    assert os.path.exists(diff_path), f"Missing diffusion at {diff_path}"

    vae, L, z_dim = load_vae(vae_path, device)
    assert z_dim == 2, "This script assumes latent_dim=2 for plotting."
    print(f"Loaded VAE: L={L}, z_dim={z_dim}")

    eps_model, betas, z_mean, z_std, T = load_latent_diffusion(diff_path, device)
    print(f"Loaded latent diffusion: T={T}, betas in [{betas.min():.2e}, {betas.max():.2e}]")

    assert X.shape[1] == L, f"Dataset L={X.shape[1]} but VAE expects L={L}"

    # plotting dirs
    plot_root = os.path.join(outdir, f"plots_dgbo_tau{args.tau_guidance}")
    step_dir = os.path.join(plot_root, "steps")
    ensure_dir(step_dir)

    # background latent cloud (dataset)
    Z_data = encode_dataset_to_z(vae, X, device, n_points=args.n_data_scatter, seed=args.data_scatter_seed)

    # -------------------------
    # Initialize evaluated set
    # -------------------------
    X_obs = []
    y_obs = []
    Z_obs = []

    if args.init_mode == "data":
        # take good seeds from dataset (highest y in dataset)
        top_idx = np.argsort(y_data)[-max(args.n_init * 8, 200):]
        init_idx = top_idx[-args.n_init:]
        for idx in init_idx:
            x = X[idx]
            yv = float(y_data[idx])
            X_obs.append(x.copy())
            y_obs.append(yv)
            Z_obs.append(encode_x_to_zmu(vae, x, device))
    elif args.init_mode == "vae_prior":
        # sample z ~ N(0,I), decode, oracle
        for _ in range(args.n_init):
            z = rng.standard_normal(2)
            x = decode_z_to_x_radians(vae, z, device)
            yv = float(oracle_f(x, target, step_size, w_close, w_smooth))
            X_obs.append(x.copy())
            y_obs.append(yv)
            Z_obs.append(encode_x_to_zmu(vae, x, device))
    else:  # diffusion init (unconditional sampling, no guidance)
        # sample z0_norm via reverse diffusion with guidance_scale=0, decode
        for _ in range(args.n_init):
            z0n, _ = guided_sample_latent_z0_norm(
                eps_model, vae, gp=None,  # unused when guidance_scale=0
                betas=betas, z_mean=z_mean, z_std=z_std,
                best_y=0.0, xi=args.xi, weight="pi",
                guidance_scale=0.0, guide_every=999999, clip_guidance=0.0,
                device=device, tau_guidance=args.tau_guidance, keep_traj=False
            )
            z0_real = (z0n.reshape(1,2) * z_std + z_mean).reshape(-1)
            x = decode_z_to_x_radians(vae, z0_real, device)
            yv = float(oracle_f(x, target, step_size, w_close, w_smooth))
            X_obs.append(x.copy())
            y_obs.append(yv)
            Z_obs.append(encode_x_to_zmu(vae, x, device))

    X_obs = np.array(X_obs, dtype=np.float64)
    y_obs = np.array(y_obs, dtype=np.float64)
    Z_obs = np.array(Z_obs, dtype=np.float64)

    print("[INIT] best y:", float(y_obs.max()))
    best_so_far = [float(y_obs.max())]

    # -------------------------
    # DGBO iterations
    # -------------------------
    for it in range(1, args.n_steps + 1):
        # cap training set size to keep GP stable
        if X_obs.shape[0] > args.max_train_gp:
            # keep best half + recent half
            n = X_obs.shape[0]
            best_idx = np.argsort(y_obs)[-args.max_train_gp//2:]
            recent_idx = np.arange(max(0, n - args.max_train_gp//2), n)
            keep = np.unique(np.concatenate([best_idx, recent_idx]))
            X_fit = X_obs[keep]
            y_fit = y_obs[keep]
        else:
            X_fit = X_obs
            y_fit = y_obs

        best_y = float(y_fit.max())
        best_all = float(y_obs.max())
        z_best = Z_obs[int(np.argmax(y_obs))]

        # fit GP in sequence space with your kernel choice
        kernel_x = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(L), nu=2.5, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel_x,
            normalize_y=True,
            n_restarts_optimizer=args.n_restarts,
            random_state=args.seed
        )
        gp.fit(X_fit, y_fit)

        # --- Acquisition grid in latent space (for plotting like COWBOYS) ---
        z1 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        z2 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        grid_x, grid_y = np.meshgrid(z1, z2)
        Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (G,2)

        # decode latent grid -> X grid in radians
        X_grid = decode_batch_z_to_x_radians(vae, Z_grid, device)    # (G,L)

        # GP predictions in x-space
        mu_g, std_g = gp.predict(X_grid, return_std=True)

        # choose what to show in the heatmap
        if args.plot_acq == "mu":
            A = mu_g
            A_name = r"$\mu(x=h(z))$"
        elif args.plot_acq == "pi":
            u = (mu_g - best_y - args.xi) / np.maximum(std_g, 1e-10)
            A = norm_cdf(u)
            A_name = r"$\mathrm{PI}(x=h(z))$"
        else:  # "ei"
            A = expected_improvement(mu_g, std_g, best_y=best_y, xi=args.xi)
            A_name = r"$\mathrm{EI}(x=h(z))$"

        A_grid = A.reshape(args.grid_res, args.grid_res)


        if it == 1 or it % 5 == 0:
            try:
                sigma_f2, ell, sigma_n2 = parse_kernel_params(gp)
                print(f"[GP] it={it:03d} | sigma_f2={sigma_f2:.3g} | noise={sigma_n2:.3g} | ell_med={np.median(ell):.3g}")
            except Exception as e:
                print(f"[GP] it={it:03d} | kernel parse failed: {e}")

        # generate guided candidates (batch)
        cand_x = []
        cand_zmu = []
        cand_score = []
        one_traj = None

        for k in range(args.n_cand):
            z0n, traj = guided_sample_latent_z0_norm(
                eps_model=eps_model,
                vae=vae,
                gp=gp,
                betas=betas,
                z_mean=z_mean,
                z_std=z_std,
                best_y=best_y,
                xi=args.xi,
                weight=args.weight,
                guidance_scale=args.guidance_scale,
                guide_every=args.guide_every,
                clip_guidance=args.clip_guidance,
                device=device,
                tau_guidance=args.tau_guidance,
                keep_traj=(k == 0)  # keep trajectory for first candidate for plotting
            )
            if k == 0:
                one_traj = traj

            z0_real = (z0n.reshape(1,2) * z_std + z_mean).reshape(-1)
            x = decode_z_to_x_radians(vae, z0_real, device)
            zmu = encode_x_to_zmu(vae, x, device)

            s = gp_acquisition(gp, x, best_y=best_y, acq=args.select_acq, xi=args.xi)

            cand_x.append(x)
            cand_zmu.append(zmu)
            cand_score.append(s)

        cand_x = np.array(cand_x, dtype=np.float64)
        cand_zmu = np.array(cand_zmu, dtype=np.float64)
        cand_score = np.array(cand_score, dtype=np.float64)

        # pick next candidate from batch
        j = int(np.argmax(cand_score))
        x_next = cand_x[j]
        z_next = cand_zmu[j]
        y_next = float(oracle_f(x_next, target, step_size, w_close, w_smooth))

        # update
        X_obs = np.vstack([X_obs, x_next.reshape(1,-1)])
        y_obs = np.append(y_obs, y_next)
        Z_obs = np.vstack([Z_obs, z_next.reshape(1,-1)])

        best_so_far.append(float(y_obs.max()))

        if it == 1 or it % 5 == 0:
            print(f"[DGBO] it={it:03d} | y_next={y_next: .4f} | best={float(y_obs.max()): .4f} "
                  f"| batch_best_{args.select_acq}={float(cand_score.max()):.3g}")

        # step plot
        save_path = os.path.join(step_dir, f"step_{it:03d}.png")
        best_idx = int(np.argmax(y_obs))
        z_best = Z_obs[best_idx]
        y_best = float(y_obs[best_idx])

        plot_step(
            save_path=save_path,
            Z_data=Z_data,
            Z_obs=Z_obs,
            y_obs=y_obs,
            Z_cand=cand_zmu,
            z_next=z_next,
            y_next=y_next,
            z_best=z_best,
            y_best=y_best,
            x_next=x_next,
            step_size=step_size,
            best_so_far=best_so_far,
            grid_x=grid_x, grid_y=grid_y, A_grid=A_grid, A_name=A_name,
            traj=one_traj
        )


    # final summary
    best_idx = int(np.argmax(y_obs))
    x_best = X_obs[best_idx]
    z_best = Z_obs[best_idx]
    y_best = float(y_obs[best_idx])

    final_png = os.path.join(plot_root, "final_summary_dgbo.png")
    plot_final_summary(final_png, Z_data, Z_obs, y_obs, z_best, x_best, y_best, step_size)

    trace_path = os.path.join(plot_root, "trace_dgbo.npz")
    np.savez_compressed(trace_path, X_obs=X_obs, y_obs=y_obs, Z_obs=Z_obs, best_so_far=np.array(best_so_far))

    print("\n=== DGBO finished ===")
    print("Best oracle:", y_best)
    print("Best z(mu):", z_best)
    print("Saved step frames:", os.path.abspath(step_dir))
    print("Saved final summary:", os.path.abspath(final_png))
    print("Saved trace:", os.path.abspath(trace_path))

if __name__ == "__main__":
    main()
