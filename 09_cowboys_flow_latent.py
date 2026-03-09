import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from toy_common import (
    VAE,
    ensure_dir,
    expected_improvement,
    load_config,
    norm_cdf,
    oracle_f,
    set_seed,
    turtle_path,
)


def fs_path(path: str) -> str:
    path = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return path
    path = os.path.normpath(path)
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def save_figure(save_path: str, dpi: int):
    ensure_dir(fs_path(os.path.dirname(save_path)))
    plt.savefig(fs_path(save_path), dpi=dpi)


def torch_load_compat(path: str, device):
    try:
        return torch.load(fs_path(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(fs_path(path), map_location=device)


def load_vae_safe(vae_path: str, device):
    ckpt = torch_load_compat(vae_path, device)
    vae = VAE(
        L=int(ckpt["L"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden=int(ckpt.get("hidden", 512)),
        enc_layers=int(ckpt.get("enc_layers", 6)),
        dec_layers=int(ckpt.get("dec_layers", 6)),
        dropout=float(ckpt.get("dropout", 0.05)),
    ).to(device)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.eval()
    return vae, int(ckpt["L"]), int(ckpt["latent_dim"]), ckpt.get("train_types", None)


@torch.no_grad()
def encode_batch_mu(vae, x_radians: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    x_radians = np.asarray(x_radians, dtype=np.float64)
    if x_radians.size == 0:
        return np.zeros((0, vae.latent_dim), dtype=np.float64)
    x_scaled = (x_radians / np.pi).astype(np.float32)
    xt = torch.from_numpy(x_scaled).to(device)
    chunks = []
    for i in range(0, xt.shape[0], batch_size):
        mu, _ = vae.encode(xt[i : i + batch_size])
        chunks.append(mu.detach().cpu().numpy())
    return np.vstack(chunks).astype(np.float64)


@torch.no_grad()
def decode_batch_z_to_x(vae, z_np: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    z_np = np.asarray(z_np, dtype=np.float64)
    if z_np.size == 0:
        return np.zeros((0, vae.L), dtype=np.float64)
    z_np = z_np.reshape(-1, vae.latent_dim)
    zt = torch.from_numpy(z_np.astype(np.float32)).to(device)
    chunks = []
    for i in range(0, zt.shape[0], batch_size):
        x_scaled = vae.decode(zt[i : i + batch_size])
        chunks.append(x_scaled.detach().cpu().numpy())
    return np.vstack(chunks).astype(np.float64) * np.pi


def ucb(mu, sigma, kappa: float = 2.0):
    return np.asarray(mu, dtype=np.float64) + float(kappa) * np.asarray(sigma, dtype=np.float64)


def probability_of_improvement(mu, sigma, best_y: float, xi: float = 0.0):
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    z = (np.asarray(mu, dtype=np.float64) - float(best_y) - float(xi)) / sigma
    return norm_cdf(z)


def gp_acq(gp, x_query: np.ndarray, best_y: float, acq_name: str, xi: float, kappa: float):
    mu, sigma = gp.predict(np.asarray(x_query, dtype=np.float64), return_std=True)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    if acq_name == "ei":
        return expected_improvement(mu, sigma, best_y=best_y, xi=xi)
    if acq_name == "pi":
        return probability_of_improvement(mu, sigma, best_y=best_y, xi=xi)
    if acq_name == "ucb":
        return ucb(mu, sigma, kappa=kappa)
    raise ValueError("acq_name must be one of: ei, pi, ucb")


# log w_t(x) used both in the latent target and in the tempered flow-training weights.
def acquisition_log_tilt(acq_values: np.ndarray, args, scale: float = 1.0) -> np.ndarray:
    acq_values = np.asarray(acq_values, dtype=np.float64)
    out = np.full(acq_values.shape, -np.inf, dtype=np.float64)
    mask = np.isfinite(acq_values)
    if args.tilt_form == "exp":
        out[mask] = float(scale) * float(args.beta_tilt) * acq_values[mask]
        return out
    if args.tilt_form == "power":
        if args.weight not in {"ei", "pi"}:
            raise ValueError("tilt_form='power' is only defined for weight in {'ei', 'pi'}.")
        out[mask] = float(scale) * float(args.beta_tilt) * np.log(np.maximum(acq_values[mask], 1e-12))
        return out
    raise ValueError("tilt_form must be one of: exp, power")


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.zeros_like(logits, dtype=np.float64)
    mask = np.isfinite(logits)
    if not np.any(mask):
        return np.full_like(logits, 1.0 / max(1, logits.size), dtype=np.float64)
    center = np.max(logits[mask])
    weights[mask] = np.exp(logits[mask] - center)
    denom = float(np.sum(weights))
    if (not np.isfinite(denom)) or denom <= 0.0:
        return np.full_like(logits, 1.0 / max(1, logits.size), dtype=np.float64)
    return weights / denom


def log_standard_normal(z: np.ndarray):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        return float(-0.5 * (z.size * np.log(2.0 * np.pi) + np.dot(z, z)))
    return -0.5 * (z.shape[1] * np.log(2.0 * np.pi) + np.sum(z * z, axis=1))


def log_gaussian_isotropic(x: np.ndarray, mean: np.ndarray, sigma: float):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = max(float(sigma) ** 2, 1e-20)
    diff = x - mean
    return float(-0.5 * (x.size * np.log(2.0 * np.pi * var) + np.dot(diff, diff) / var))


def finite_mean(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float(default)
    return float(np.mean(arr[mask]))


def finite_std(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float(default)
    return float(np.std(arr[mask]))


class CouplingMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor, scale_limit: float = 2.0):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1))
        self.scale_net = CouplingMLP(dim, hidden)
        self.shift_net = CouplingMLP(dim, hidden)
        self.scale_limit = float(scale_limit)

    def _st(self, x_masked: torch.Tensor):
        raw_scale = self.scale_net(x_masked)
        log_scale = self.scale_limit * torch.tanh(raw_scale / max(self.scale_limit, 1e-6))
        log_scale = (1.0 - self.mask) * log_scale
        shift = (1.0 - self.mask) * self.shift_net(x_masked)
        return log_scale, shift

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask
        log_scale, shift = self._st(x_masked)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        return y, torch.sum(log_scale, dim=1)

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        log_scale, shift = self._st(y_masked)
        x = y_masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        return x, -torch.sum(log_scale, dim=1)


class RealNVP(nn.Module):
    def __init__(self, dim: int, hidden: int, depth: int):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.depth = int(depth)

        base_mask = (torch.arange(self.dim) % 2 == 0).float()
        layers = []
        for i in range(self.depth):
            mask = base_mask if (i % 2 == 0) else (1.0 - base_mask)
            layers.append(AffineCoupling(self.dim, self.hidden, mask))
        self.layers = nn.ModuleList(layers)

    def _base_log_prob(self, z: torch.Tensor):
        return -0.5 * (self.dim * np.log(2.0 * np.pi) + torch.sum(z * z, dim=1))

    def forward_transform(self, z: torch.Tensor):
        x = z
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse_transform(self, x: torch.Tensor):
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x: torch.Tensor):
        z, log_det = self.inverse_transform(x)
        return self._base_log_prob(z) + log_det

    def sample(self, n: int):
        device = next(self.parameters()).device
        z = torch.randn(int(n), self.dim, device=device)
        x, _ = self.forward_transform(z)
        return x


def build_flow(flow_type: str, latent_dim: int, hidden: int, depth: int, device):
    if flow_type != "realnvp":
        raise ValueError("Only flow_type='realnvp' is implemented in this script.")
    flow = RealNVP(dim=latent_dim, hidden=hidden, depth=depth).to(device)
    flow.eval()
    return flow


def save_flow_checkpoint(path: str, flow, args, latent_dim: int, flow_loss_hist):
    ckpt = {
        "state_dict": flow.state_dict(),
        "flow_type": args.flow_type,
        "latent_dim": int(latent_dim),
        "flow_hidden": int(args.flow_hidden),
        "flow_depth": int(args.flow_depth),
        "flow_loss_last_hist": np.asarray(flow_loss_hist, dtype=np.float32),
    }
    ensure_dir(fs_path(os.path.dirname(path)))
    torch.save(ckpt, fs_path(path))


def load_flow_checkpoint(path: str, flow, device):
    ckpt = torch_load_compat(path, device)
    flow.load_state_dict(ckpt["state_dict"], strict=True)
    flow.eval()
    return ckpt


@torch.no_grad()
def flow_log_prob_np(flow, z_np: np.ndarray, device) -> np.ndarray:
    z_np = np.asarray(z_np, dtype=np.float64)
    if z_np.ndim == 1:
        z_np = z_np.reshape(1, -1)
    zt = torch.from_numpy(z_np.astype(np.float32)).to(device)
    lp = flow.log_prob(zt).detach().cpu().numpy().astype(np.float64)
    return lp


@torch.no_grad()
def sample_flow_np(flow, n: int, device) -> np.ndarray:
    z = flow.sample(int(n))
    return z.detach().cpu().numpy().astype(np.float64)


def train_flow_weighted(flow, z_pool: np.ndarray, weights: np.ndarray, args, device, n_steps=None, trust_coef=None):
    z_pool = np.asarray(z_pool, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    zt = torch.from_numpy(z_pool.astype(np.float32)).to(device)
    wt = torch.from_numpy(weights.astype(np.float32)).to(device)
    train_steps = int(args.flow_train_steps if n_steps is None else n_steps)
    trust = float(args.flow_trust_coef if trust_coef is None else trust_coef)

    if zt.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.nan

    ref_params = None
    if trust > 0.0:
        ref_params = [p.detach().clone() for p in flow.parameters()]

    # Weighted MLE adapts q_theta(z) to a tempered acquisition-tilted pool; q_theta stays proposal-only.
    opt = optim.AdamW(flow.parameters(), lr=args.flow_lr, weight_decay=args.flow_weight_decay)
    batch_size = min(max(1, int(args.flow_batch)), zt.shape[0])
    loss_hist = []

    flow.train()
    for _ in range(train_steps):
        if batch_size >= zt.shape[0]:
            loss = -(wt * flow.log_prob(zt)).sum()
        else:
            idx = torch.multinomial(wt, num_samples=batch_size, replacement=True)
            zb = zt[idx]
            loss = -flow.log_prob(zb).mean()

        if ref_params is not None:
            penalty = torch.zeros((), dtype=torch.float32, device=device)
            for p, p_ref in zip(flow.parameters(), ref_params):
                penalty = penalty + torch.mean((p - p_ref) ** 2)
            loss = loss + trust * penalty

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.flow_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(args.flow_grad_clip))
        opt.step()
        loss_hist.append(float(loss.item()))

    flow.eval()
    with torch.no_grad():
        full_loss = float((-(wt * flow.log_prob(zt)).sum()).item())
    return np.asarray(loss_hist, dtype=np.float64), full_loss


@torch.no_grad()
def estimate_flow_entropy(flow, n_samples: int, device):
    if n_samples <= 0:
        return np.nan
    z = flow.sample(int(n_samples))
    lp = flow.log_prob(z)
    return float((-lp.mean()).item())


def sample_local_cloud(centers: np.ndarray, n_total: int, sigma: float, rng: np.random.Generator):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0 or n_total <= 0:
        return np.zeros((0, centers.shape[1] if centers.ndim == 2 else 0), dtype=np.float64)
    idx = rng.integers(0, centers.shape[0], size=int(n_total))
    noise = float(sigma) * rng.standard_normal(size=(int(n_total), centers.shape[1]))
    return centers[idx] + noise


def normalize_mixture_weights(weights, label: str) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0.0):
        raise ValueError(f"{label} weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{label} weights must sum to a positive value.")
    return weights / total


def fit_flow_to_latent_codes(z_codes: np.ndarray, args, device):
    z_codes = np.asarray(z_codes, dtype=np.float64)
    if z_codes.ndim != 2 or z_codes.shape[0] == 0:
        raise ValueError("Need at least one latent code to train the initialization flow.")
    init_flow = build_flow(args.flow_type, z_codes.shape[1], args.flow_hidden, args.flow_depth, device)
    weights = np.full(z_codes.shape[0], 1.0 / z_codes.shape[0], dtype=np.float64)
    loss_hist, loss_last = train_flow_weighted(
        init_flow,
        z_codes,
        weights,
        args,
        device,
        n_steps=args.init_flow_train_steps,
        trust_coef=0.0,
    )
    return init_flow, loss_hist, float(loss_last)


def sample_initial_latents(args, vae, x_data: np.ndarray, latent_dim: int, rng: np.random.Generator, device):
    n_init = int(args.n_init)
    if args.init_mode == "prior":
        z_init = rng.standard_normal(size=(n_init, latent_dim))
        return z_init.astype(np.float64), {
            "mode": args.init_mode,
            "counts": {"flow": 0, "prior": n_init, "local": 0},
            "flow_loss_last": np.nan,
            "flow_loss_mean": np.nan,
            "n_data_codes": 0,
        }

    z_data = encode_batch_mu(vae, x_data, device)
    if z_data.shape[0] == 0:
        raise ValueError("The dataset produced no latent codes for initialization.")

    counts = {"flow": 0, "prior": 0, "local": 0}
    need_flow = args.init_mode == "flow"
    if args.init_mode == "flow":
        counts["flow"] = n_init
    elif args.init_mode == "mix":
        probs = normalize_mixture_weights(
            np.array([args.init_mix_flow, args.init_mix_prior, args.init_mix_local], dtype=np.float64),
            label="Initialization mixture",
        )
        draw = rng.multinomial(n_init, probs)
        counts["flow"] = int(draw[0])
        counts["prior"] = int(draw[1])
        counts["local"] = int(draw[2])
        need_flow = counts["flow"] > 0
    else:
        raise ValueError("init_mode must be one of: prior, flow, mix")

    init_flow = None
    init_flow_loss_hist = np.zeros((0,), dtype=np.float64)
    init_flow_loss_last = np.nan
    if need_flow:
        init_flow, init_flow_loss_hist, init_flow_loss_last = fit_flow_to_latent_codes(z_data, args, device)

    parts = []
    if counts["flow"] > 0:
        parts.append(sample_flow_np(init_flow, counts["flow"], device))
    if counts["prior"] > 0:
        parts.append(rng.standard_normal(size=(counts["prior"], latent_dim)))
    if counts["local"] > 0:
        parts.append(sample_local_cloud(z_data, counts["local"], float(args.init_local_scale), rng))

    z_init = np.vstack(parts) if parts else np.zeros((0, latent_dim), dtype=np.float64)
    if z_init.shape[0] > 1:
        z_init = z_init[rng.permutation(z_init.shape[0])]

    return z_init.astype(np.float64), {
        "mode": args.init_mode,
        "counts": counts,
        "flow_loss_last": float(init_flow_loss_last),
        "flow_loss_mean": finite_mean(init_flow_loss_hist, default=np.nan),
        "n_data_codes": int(z_data.shape[0]),
    }


def build_latent_pool(z_obs_mu, y_obs, replay_buffer, args, rng, latent_dim):
    pieces = []
    labels = []

    if z_obs_mu.shape[0] > 0:
        pieces.append(z_obs_mu)
        labels.extend(["obs"] * z_obs_mu.shape[0])

    if replay_buffer.shape[0] > 0 and args.pool_replay > 0:
        take = min(int(args.pool_replay), replay_buffer.shape[0])
        idx = rng.choice(replay_buffer.shape[0], size=take, replace=False)
        z_rep = replay_buffer[idx]
        pieces.append(z_rep)
        labels.extend(["replay"] * z_rep.shape[0])

    if args.pool_prior > 0:
        z_prior = rng.standard_normal(size=(int(args.pool_prior), latent_dim))
        pieces.append(z_prior)
        labels.extend(["prior"] * z_prior.shape[0])

    if args.pool_local > 0 and z_obs_mu.shape[0] > 0:
        k = min(int(args.pool_topk), z_obs_mu.shape[0])
        top_idx = np.argsort(y_obs)[-k:]
        centers = z_obs_mu[top_idx]
        z_local = sample_local_cloud(centers, int(args.pool_local), float(args.pool_local_scale), rng)
        pieces.append(z_local)
        labels.extend(["local"] * z_local.shape[0])

    z_pool = np.vstack(pieces).astype(np.float64)
    return z_pool, np.asarray(labels)


def append_replay_buffer(replay_buffer: np.ndarray, z_new: np.ndarray, max_size: int):
    z_new = np.asarray(z_new, dtype=np.float64)
    if z_new.size == 0:
        return replay_buffer
    if replay_buffer.size == 0:
        replay = z_new.copy()
    else:
        replay = np.vstack([replay_buffer, z_new])
    if replay.shape[0] > int(max_size):
        replay = replay[-int(max_size) :]
    return replay


def latent_grid_acq_2d(vae, gp, best_y: float, args, zlim, device):
    z1 = np.linspace(zlim[0][0], zlim[0][1], int(args.grid_res))
    z2 = np.linspace(zlim[1][0], zlim[1][1], int(args.grid_res))
    gx, gy = np.meshgrid(z1, z2)
    zg = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xg = decode_batch_z_to_x(vae, zg, device)
    ag = gp_acq(gp, xg, best_y, args.plot_acq, args.xi, args.kappa).reshape(int(args.grid_res), int(args.grid_res))
    title = {"ei": "EI(x)", "pi": "PI(x)", "ucb": "UCB(x)"}[args.plot_acq]
    return gx, gy, ag, title


def resolve_latent_plot_limits(args, z_plot: np.ndarray):
    if args.latent_xlim is not None and args.latent_ylim is not None:
        return (
            (float(args.latent_xlim[0]), float(args.latent_xlim[1])),
            (float(args.latent_ylim[0]), float(args.latent_ylim[1])),
        )

    z_plot = np.asarray(z_plot, dtype=np.float64)
    qx = np.quantile(z_plot[:, 0], [0.02, 0.98])
    qy = np.quantile(z_plot[:, 1], [0.02, 0.98])
    sx = float(qx[1] - qx[0])
    sy = float(qy[1] - qy[0])
    return (
        (float(qx[0] - 0.15 * max(sx, 1e-3)), float(qx[1] + 0.15 * max(sx, 1e-3))),
        (float(qy[0] - 0.15 * max(sy, 1e-3)), float(qy[1] + 0.15 * max(sy, 1e-3))),
    )


def plot_step(
    save_path,
    z_bg,
    z_obs,
    y_obs,
    z_next,
    z_best,
    z_cand,
    x_next,
    y_next,
    step_size,
    best_so_far,
    zlim,
    gx,
    gy,
    a_grid,
    a_name,
):
    fig = plt.figure(figsize=(14.6, 5.0))

    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(gx, gy, a_grid, levels=35)
    plt.colorbar(im, ax=ax1, label=a_name)
    ax1.scatter(z_bg[:, 0], z_bg[:, 1], s=8, alpha=0.10, linewidth=0.0, label="data")
    ax1.scatter(z_obs[:, 0], z_obs[:, 1], c=y_obs, s=52, edgecolor="black", linewidth=0.35, label="evaluated")
    #if z_cand.shape[0] > 0:
    #    ax1.scatter(z_cand[:, 0], z_cand[:, 1], s=22, alpha=0.75, marker="x", label="candidates")
    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0, label="best")
    ax1.scatter([z_next[0]], [z_next[1]], c="yellow", s=150, marker="D", edgecolor="black", linewidth=1.0, label="next")
    ax1.set_title("VAE latent acquisition")
    ax1.set_xlabel("z1")
    ax1.set_ylabel("z2")
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(zlim[0])
    ax1.set_ylim(zlim[1])
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 3, 2)
    pts_next = turtle_path(x_next, step_size)
    ax2.plot(pts_next[:, 0], pts_next[:, 1], linewidth=2.4, label="x_next")
    ax2.scatter([pts_next[0, 0]], [pts_next[0, 1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Decoded x_next\ny={y_next:.3f}")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(best_so_far, marker="o", linewidth=1.4, label="best-so-far y")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("best oracle")
    ax3.set_title("Optimization progress")
    ax3.legend(loc="best")

    plt.tight_layout()
    save_figure(save_path, dpi=170)
    plt.close(fig)


def plot_final(
    save_path,
    z_bg,
    z_obs,
    y_obs,
    z_best,
    x_best,
    y_best,
    step_size,
    best_so_far,
    acc_hist,
    zlim,
):
    fig = plt.figure(figsize=(16.0, 5.0))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(z_bg[:, 0], z_bg[:, 1], s=8, alpha=0.12, linewidth=0.0, label="data")
    sc = ax1.scatter(z_obs[:, 0], z_obs[:, 1], c=y_obs, s=52, edgecolor="black", linewidth=0.35, label="BO points")
    plt.colorbar(sc, ax=ax1, label="oracle y")
    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0, label="best")
    ax1.set_title("Final VAE latent overlay")
    ax1.set_xlabel("z1")
    ax1.set_ylabel("z2")
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(zlim[0])
    ax1.set_ylim(zlim[1])
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 4, 2)
    pts_best = turtle_path(x_best, step_size)
    ax2.plot(pts_best[:, 0], pts_best[:, 1], linewidth=2.4, label="best")
    ax2.scatter([pts_best[0, 0]], [pts_best[0, 1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Best decoded path\ny={y_best:.3f}")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(x_best)
    ax3.axhline(0.0, linestyle=":", linewidth=1)
    ax3.set_title("Best sequence x")
    ax3.set_xlabel("t")
    ax3.set_ylabel("delta theta")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(best_so_far, marker="o", linewidth=1.4, label="best-so-far y")
    ax4_t = ax4.twinx()
    ax4_t.plot(acc_hist, color="tab:orange", marker="x", linewidth=1.3, label="MH accept")
    ax4.set_xlabel("iteration")
    ax4.set_title("Optimization traces")
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4_t.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    save_figure(save_path, dpi=180)
    plt.close(fig)


def plot_latent_flow_diag(
    save_path,
    z_bg,
    z_pool,
    pool_weights,
    z_flow,
    z_prior,
    z_replay,
    z_chain,
    iteration: int,
):
    fig = plt.figure(figsize=(13.6, 5.2))

    ax1 = fig.add_subplot(1, 2, 1)
    if z_bg.shape[0] > 0:
        ax1.scatter(z_bg[:, 0], z_bg[:, 1], s=8, alpha=0.08, linewidth=0.0, label="encoded data")
    if z_replay.shape[0] > 0:
        ax1.scatter(z_replay[:, 0], z_replay[:, 1], s=10, alpha=0.18, linewidth=0.0, label="replay")
    if z_pool.shape[0] > 0:
        sc = ax1.scatter(
            z_pool[:, 0],
            z_pool[:, 1],
            c=pool_weights,
            s=28,
            alpha=0.85,
            linewidth=0.0,
            cmap="viridis",
            label="latent pool",
        )
        plt.colorbar(sc, ax=ax1, label="proposal train weight")
    if z_flow.shape[0] > 0:
        ax1.scatter(z_flow[:, 0], z_flow[:, 1], s=22, alpha=0.65, marker="x", label="flow samples")
    if z_chain.shape[0] > 1:
        ax1.plot(z_chain[:, 0], z_chain[:, 1], color="tab:green", linewidth=1.1, alpha=0.9, label="MH chain")
    ax1.set_title(f"Latent pool vs flow proposal (iter {iteration:03d})")
    ax1.set_xlabel("z1")
    ax1.set_ylabel("z2")
    ax1.set_aspect("equal", "box")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 2, 2)
    if z_prior.shape[0] > 0:
        ax2.scatter(z_prior[:, 0], z_prior[:, 1], s=12, alpha=0.22, linewidth=0.0, label="prior N(0, I)")
    if z_flow.shape[0] > 0:
        ax2.scatter(z_flow[:, 0], z_flow[:, 1], s=22, alpha=0.65, marker="x", label="flow")
    ax2.set_title("Prior vs flow samples")
    ax2.set_xlabel("z1")
    ax2.set_ylabel("z2")
    ax2.set_aspect("equal", "box")
    ax2.legend(loc="best")

    plt.tight_layout()
    save_figure(save_path, dpi=170)
    plt.close(fig)


def plot_flow_training_overview(save_path, flow_loss_last_hist, flow_loss_mean_hist, entropy_hist):
    if len(flow_loss_last_hist) == 0:
        return
    fig = plt.figure(figsize=(11.0, 4.5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(flow_loss_last_hist, marker="o", linewidth=1.4, label="final weighted NLL")
    ax1.plot(flow_loss_mean_hist, marker="x", linewidth=1.2, label="mean train loss")
    ax1.set_xlabel("iteration")
    ax1.set_title("Flow training loss by BO iteration")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(entropy_hist, marker="o", linewidth=1.4)
    ax2.set_xlabel("iteration")
    ax2.set_title("Proposal entropy estimate")
    ax2.set_ylabel("-E_q log q")

    plt.tight_layout()
    save_figure(save_path, dpi=170)
    plt.close(fig)


def validate_args(args):
    if args.n_init <= 0:
        raise ValueError("--n_init must be > 0.")
    if args.beta_tilt <= 0.0:
        raise ValueError("--beta_tilt must be > 0.")
    if args.tau_temp <= 0.0:
        raise ValueError("--tau_temp must be > 0.")
    if not (0.0 <= args.lambda_local <= 1.0):
        raise ValueError("--lambda_local must lie in [0, 1].")
    if args.sigma_local <= 0.0:
        raise ValueError("--sigma_local must be > 0.")
    if args.mh_steps <= 0 or args.mh_burn < 0 or args.mh_thin <= 0:
        raise ValueError("--mh_steps > 0, --mh_burn >= 0, and --mh_thin > 0 are required.")
    if args.flow_hidden <= 0 or args.flow_depth <= 0:
        raise ValueError("--flow_hidden and --flow_depth must be positive.")
    if args.flow_batch <= 0 or args.flow_train_steps < 0:
        raise ValueError("--flow_batch must be > 0 and --flow_train_steps must be >= 0.")
    if args.init_flow_train_steps < 0:
        raise ValueError("--init_flow_train_steps must be >= 0.")
    if args.init_local_scale <= 0.0:
        raise ValueError("--init_local_scale must be > 0.")
    if min(args.init_mix_flow, args.init_mix_prior, args.init_mix_local) < 0.0:
        raise ValueError("--init_mix_flow, --init_mix_prior, and --init_mix_local must be non-negative.")
    if args.init_mode == "flow" and args.init_flow_train_steps <= 0:
        raise ValueError("--init_flow_train_steps must be > 0 when --init_mode=flow.")
    if args.init_mode == "mix":
        mix_total = float(args.init_mix_flow + args.init_mix_prior + args.init_mix_local)
        if mix_total <= 0.0:
            raise ValueError("The initialization mixture weights must sum to a positive value when --init_mode=mix.")
        if args.init_mix_flow > 0.0 and args.init_flow_train_steps <= 0:
            raise ValueError("--init_flow_train_steps must be > 0 when the mix includes flow samples.")
    if (args.latent_xlim is None) != (args.latent_ylim is None):
        raise ValueError("--latent_xlim and --latent_ylim must be provided together.")
    if args.latent_xlim is not None and float(args.latent_xlim[0]) >= float(args.latent_xlim[1]):
        raise ValueError("--latent_xlim must satisfy min < max.")
    if args.latent_ylim is not None and float(args.latent_ylim[0]) >= float(args.latent_ylim[1]):
        raise ValueError("--latent_ylim must satisfy min < max.")
    if args.weight == "ucb" and args.tilt_form == "power":
        raise ValueError("tilt_form='power' is not valid when --weight=ucb.")


# Deterministic-decoder latent target: log pi_t(z) = log p(z) + log w_t(g(z)).
def evaluate_latent_target(z: np.ndarray, vae, gp, best_y: float, args, device):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(z)):
        return -np.inf, np.full((vae.L,), np.nan, dtype=np.float64), np.nan
    x = decode_batch_z_to_x(vae, z.reshape(1, -1), device)[0]
    acq = float(gp_acq(gp, x.reshape(1, -1), best_y, args.weight, args.xi, args.kappa)[0])
    log_tilt = float(acquisition_log_tilt(np.array([acq], dtype=np.float64), args)[0])
    log_prior = float(log_standard_normal(z))
    log_pi = log_prior + log_tilt
    if not np.isfinite(log_pi):
        return -np.inf, x, acq
    return float(log_pi), x, acq


# Exact mixture density used in the MH accept ratio.
def log_q_mix(z_to: np.ndarray, z_from: np.ndarray, log_q_flow_to: float, args):
    lam = float(args.lambda_local)
    if lam <= 0.0:
        return float(log_q_flow_to)
    if lam >= 1.0:
        return log_gaussian_isotropic(z_to, z_from, args.sigma_local)
    log_local = log_gaussian_isotropic(z_to, z_from, args.sigma_local)
    return float(np.logaddexp(np.log(lam) + log_local, np.log(1.0 - lam) + float(log_q_flow_to)))


def choose_chain_init(args, z_best_enc, replay_buffer, latent_dim: int, rng: np.random.Generator):
    if args.chain_init == "best":
        return z_best_enc.copy()
    if args.chain_init == "prior":
        return rng.standard_normal(size=(latent_dim,))
    if args.chain_init == "replay":
        if replay_buffer.shape[0] > 0:
            return replay_buffer[rng.integers(0, replay_buffer.shape[0])].copy()
        return z_best_enc.copy()
    raise ValueError("chain_init must be one of: best, prior, replay")


def make_valid_init(z0, vae, gp, best_y: float, args, flow, device, rng):
    d = z0.shape[0]
    candidates = [np.asarray(z0, dtype=np.float64).copy()]
    for _ in range(16):
        candidates.append(rng.standard_normal(size=(d,)))
    for z in candidates:
        log_pi, x, acq = evaluate_latent_target(z, vae, gp, best_y, args, device)
        if np.isfinite(log_pi):
            log_q = float(flow_log_prob_np(flow, z, device)[0])
            return z, log_pi, x, acq, log_q
    z = candidates[0]
    log_pi, x, acq = evaluate_latent_target(z, vae, gp, best_y, args, device)
    log_q = float(flow_log_prob_np(flow, z, device)[0])
    return z, log_pi, x, acq, log_q


def mh_sample_latent_candidates(z0, flow, vae, gp, best_y: float, args, rng, device):
    z, log_pi_z, x_z, _, log_q_flow_z = make_valid_init(z0, vae, gp, best_y, args, flow, device, rng)
    d = z.shape[0]

    accepted = 0
    n_flow = 0
    n_local = 0
    z_samples = []
    x_samples = []
    z_chain = []
    x_chain = []
    chain_logpi = []
    accepted_states = []
    log_alpha_trace = []
    jump_norms = []

    post_burn_counter = 0

    for step in range(1, int(args.mh_steps) + 1):
        z_prev = z.copy()
        if rng.random() < float(args.lambda_local):
            n_local += 1
            z_prop = z_prev + float(args.sigma_local) * rng.standard_normal(size=(d,))
        else:
            n_flow += 1
            z_prop = sample_flow_np(flow, 1, device)[0]

        jump_norms.append(float(np.linalg.norm(z_prop - z_prev)))
        log_q_flow_prop = float(flow_log_prob_np(flow, z_prop, device)[0])
        log_pi_prop, x_prop, _ = evaluate_latent_target(z_prop, vae, gp, best_y, args, device)

        log_q_fwd = log_q_mix(z_prop, z_prev, log_q_flow_prop, args)
        log_q_rev = log_q_mix(z_prev, z_prop, log_q_flow_z, args)

        if not np.isfinite(log_pi_z):
            log_alpha = 0.0 if np.isfinite(log_pi_prop) else -np.inf
        elif np.isfinite(log_pi_prop) and np.isfinite(log_q_fwd) and np.isfinite(log_q_rev):
            log_alpha = log_pi_prop + log_q_rev - log_pi_z - log_q_fwd
        else:
            log_alpha = -np.inf

        if np.log(rng.random()) < min(0.0, float(log_alpha)):
            z = z_prop
            x_z = x_prop
            log_pi_z = log_pi_prop
            log_q_flow_z = log_q_flow_prop
            accepted += 1
            accepted_states.append(z.copy())

        log_alpha_trace.append(float(log_alpha))

        if step % int(args.mh_thin) == 0:
            z_chain.append(z.copy())
            x_chain.append(x_z.copy())
            chain_logpi.append(float(log_pi_z))

        if step > int(args.mh_burn):
            post_burn_counter += 1
            if post_burn_counter % int(args.mh_thin) == 0:
                z_samples.append(z.copy())
                x_samples.append(x_z.copy())

    if len(z_samples) == 0:
        z_samples.append(z.copy())
        x_samples.append(x_z.copy())

    z_samples = np.asarray(z_samples, dtype=np.float64)
    x_samples = np.asarray(x_samples, dtype=np.float64)
    z_chain = np.asarray(z_chain, dtype=np.float64) if z_chain else np.zeros((0, d), dtype=np.float64)
    x_chain = np.asarray(x_chain, dtype=np.float64) if x_chain else np.zeros((0, vae.L), dtype=np.float64)
    chain_logpi = np.asarray(chain_logpi, dtype=np.float64) if chain_logpi else np.zeros((0,), dtype=np.float64)
    accepted_states = np.asarray(accepted_states, dtype=np.float64) if accepted_states else np.zeros((0, d), dtype=np.float64)

    stats = {
        "accept_rate": float(accepted / max(1, int(args.mh_steps))),
        "jump_norm_mean": finite_mean(jump_norms, default=0.0),
        "log_alpha_mean": finite_mean(log_alpha_trace, default=-np.inf),
        "log_alpha_std": finite_std(log_alpha_trace, default=np.nan),
        "frac_flow_proposals_used": float(n_flow / max(1, int(args.mh_steps))),
        "frac_local_proposals_used": float(n_local / max(1, int(args.mh_steps))),
        "latent_norm_mean": finite_mean(np.linalg.norm(z_samples, axis=1), default=np.nan),
    }
    return z_samples, x_samples, z_chain, x_chain, chain_logpi, accepted_states, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--plotroot", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_init", type=int, default=24)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--init_mode", type=str, default="mix", choices=["prior", "flow", "mix"])
    ap.add_argument("--init_mix_flow", type=float, default=0.7)
    ap.add_argument("--init_mix_prior", type=float, default=0.2)
    ap.add_argument("--init_mix_local", type=float, default=0.1)
    ap.add_argument("--init_local_scale", type=float, default=0.2)
    ap.add_argument("--latent_xlim", type=float, nargs=2, default=None)
    ap.add_argument("--latent_ylim", type=float, nargs=2, default=None)

    ap.add_argument("--weight", type=str, default="pi", choices=["pi", "ei", "ucb"])
    ap.add_argument("--select_acq", type=str, default=None, choices=["pi", "ei", "ucb"])
    ap.add_argument("--plot_acq", type=str, default=None, choices=["pi", "ei", "ucb"])
    ap.add_argument("--xi", type=float, default=0.06)
    ap.add_argument("--kappa", type=float, default=2.2)
    ap.add_argument("--beta_tilt", type=float, default=8.0)
    ap.add_argument("--tau_temp", type=float, default=0.5)
    ap.add_argument("--tilt_form", type=str, default="exp", choices=["exp", "power"])
    ap.add_argument(
        "--allow_mismatched_select",
        action="store_true",
        help="Allow select_acq != weight. By default the selection acquisition is forced to match the target weight.",
    )

    ap.add_argument("--chain_init", type=str, default="best", choices=["best", "prior", "replay"])
    ap.add_argument("--mh_steps", type=int, default=1200)
    ap.add_argument("--mh_burn", type=int, default=300)
    ap.add_argument("--mh_thin", type=int, default=4)
    ap.add_argument("--lambda_local", type=float, default=0.35)
    ap.add_argument("--sigma_local", type=float, default=0.35)

    ap.add_argument("--flow_type", type=str, default="realnvp", choices=["realnvp"])
    ap.add_argument("--flow_hidden", type=int, default=192)
    ap.add_argument("--flow_depth", type=int, default=6)
    ap.add_argument("--flow_train_steps", type=int, default=700)
    ap.add_argument("--init_flow_train_steps", type=int, default=400)
    ap.add_argument("--flow_lr", type=float, default=2e-4)
    ap.add_argument("--flow_weight_decay", type=float, default=1e-5)
    ap.add_argument("--flow_batch", type=int, default=256)
    ap.add_argument("--flow_grad_clip", type=float, default=1.0)
    ap.add_argument("--flow_trust_coef", type=float, default=0.0)
    ap.add_argument("--flow_ckpt", type=str, default=None, help="Optional checkpoint to warm-start the flow.")

    ap.add_argument("--pool_prior", type=int, default=512)
    ap.add_argument("--pool_replay", type=int, default=256)
    ap.add_argument("--pool_local", type=int, default=256)
    ap.add_argument("--pool_local_scale", type=float, default=0.35)
    ap.add_argument("--pool_topk", type=int, default=6)
    ap.add_argument("--replay_max", type=int, default=5000)

    ap.add_argument("--vae_ckpt", type=str, default=None)
    ap.add_argument("--deterministic_decode", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--grid_res", type=int, default=110)
    ap.add_argument("--n_data_scatter", type=int, default=6000)
    ap.add_argument("--n_flow_diag", type=int, default=800)
    ap.add_argument("--flow_entropy_samples", type=int, default=512)
    args = ap.parse_args()

    if args.select_acq is None:
        args.select_acq = args.weight
    if args.plot_acq is None:
        args.plot_acq = args.weight
    if (not args.allow_mismatched_select) and args.select_acq != args.weight:
        print(
            f"[INFO] forcing select_acq to match weight: {args.select_acq} -> {args.weight}. "
            "Use --allow_mismatched_select to disable this."
        )
        args.select_acq = args.weight

    validate_args(args)

    if not args.deterministic_decode:
        print("[INFO] stochastic decode is not implemented for this VAE; using decoder mean.")
        args.deterministic_decode = True

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_npz = os.path.join(args.outdir, "data", "dataset.npz")
    assert os.path.exists(fs_path(data_npz)), f"Missing {data_npz}"
    data = np.load(fs_path(data_npz), allow_pickle=True)
    x_all = data["X"].astype(np.float64)
    target = data["target"].astype(np.float64)
    step_size, w_close, w_smooth = load_config(args.outdir)

    vae_ckpt = args.vae_ckpt or os.path.join(args.outdir, "models", "vae.pt")
    assert os.path.exists(fs_path(vae_ckpt)), f"Missing VAE checkpoint at {vae_ckpt}"
    vae, seq_len, latent_dim, train_types = load_vae_safe(vae_ckpt, device)
    assert x_all.shape[1] == seq_len, f"Dataset length {x_all.shape[1]} != VAE length {seq_len}"

    print(f"Loaded dataset: N={x_all.shape[0]}, L={seq_len}")
    print(f"Loaded VAE: latent_dim={latent_dim}, train_types={train_types}")
    print(
        f"[TARGET] latent pi(z) proportional to N(0, I) * w(g(z)) | weight={args.weight} "
        f"| tilt_form={args.tilt_form} | beta_tilt={args.beta_tilt}"
    )
    print(
        f"[PROPOSAL] q_mix(z'|z) = lambda_local N(z'; z, sigma^2 I) + (1-lambda_local) q_theta(z') "
        f"| flow={args.flow_type} | lambda_local={args.lambda_local} | sigma_local={args.sigma_local}"
    )
    print(
        f"[INIT-PROPOSAL] mode={args.init_mode} | flow/prior/local=({args.init_mix_flow}, "
        f"{args.init_mix_prior}, {args.init_mix_local}) | init_local_scale={args.init_local_scale}"
    )
    if args.latent_xlim is not None:
        print(f"[LATENT-PLOT] using fixed box x={tuple(args.latent_xlim)} y={tuple(args.latent_ylim)}")
    else:
        print("[LATENT-PLOT] using adaptive quantile box")

    if args.plotroot is None:
        args.plotroot = os.path.join(
            args.outdir,
            "09_cowboys_flow_latent",
            f"{args.weight}_bt{args.beta_tilt}_tau{args.tau_temp}_init{args.init_mode}_mh{args.mh_steps}",
        )
    plot_root = args.plotroot
    step_dir = os.path.join(plot_root, "steps")
    diag_dir = os.path.join(plot_root, "diagnostics")
    flow_dir = os.path.join(plot_root, "flow_ckpts")
    ensure_dir(fs_path(step_dir))
    ensure_dir(fs_path(diag_dir))
    ensure_dir(fs_path(flow_dir))

    with open(fs_path(os.path.join(plot_root, "run_config.json")), "w", encoding="utf-8") as f_cfg:
        json.dump(vars(args), f_cfg, indent=2)

    flow = build_flow(args.flow_type, latent_dim, args.flow_hidden, args.flow_depth, device)
    if args.flow_ckpt is not None and os.path.exists(fs_path(args.flow_ckpt)):
        load_flow_checkpoint(args.flow_ckpt, flow, device)
        print("[FLOW] warm-started from:", os.path.abspath(args.flow_ckpt))

    n_bg = min(int(args.n_data_scatter), x_all.shape[0])
    idx_bg = rng.choice(x_all.shape[0], size=n_bg, replace=False)
    x_bg = x_all[idx_bg]
    z_bg = encode_batch_mu(vae, x_bg, device)
    if latent_dim != 2:
        raise ValueError(f"This script now assumes a 2D VAE latent space, but latent_dim={latent_dim}.")

    z_init, init_info = sample_initial_latents(args, vae, x_all, latent_dim, rng, device)
    x_init = decode_batch_z_to_x(vae, z_init, device)
    y_init = np.array(
        [float(oracle_f(x, target, step_size, w_close, w_smooth)) for x in x_init],
        dtype=np.float64,
    )
    init_counts = init_info["counts"]
    print(
        f"[INIT] sampled n_init={z_init.shape[0]} from mode={init_info['mode']} | "
        f"flow={init_counts['flow']} prior={init_counts['prior']} local={init_counts['local']} "
        f"| data_codes={init_info['n_data_codes']} | init_flow_loss={init_info['flow_loss_last']:.4f}"
    )

    z_obs = z_init.copy()
    x_obs = x_init.copy()
    y_obs = y_init.copy()
    replay_buffer = np.zeros((0, latent_dim), dtype=np.float64)

    best_so_far = [float(np.max(y_obs))]
    acc_hist = [np.nan]
    flow_loss_last_hist = []
    flow_loss_mean_hist = []
    entropy_hist = []
    replay_size_hist = []
    low_acc_count = 0

    metrics_path = os.path.join(plot_root, "metrics.csv")
    flow_overview_path = os.path.join(diag_dir, "flow_training_overview.png")

    print("[INIT] best y:", float(np.max(y_obs)))

    with open(fs_path(metrics_path), "w", newline="", encoding="utf-8") as f_csv:
        wr = csv.writer(f_csv)
        wr.writerow(
            [
                "iter",
                "y_next",
                "best_y",
                "accept_rate",
                "cand_util_max",
                "cand_util_mean",
                "chain_logpi_last",
                "jump_norm_mean",
                "log_alpha_mean",
                "log_alpha_std",
                "flow_loss_last",
                "flow_loss_mean",
                "proposal_entropy_estimate",
                "frac_flow_proposals_used",
                "frac_local_proposals_used",
                "latent_norm_mean",
                "pool_size",
                "replay_size",
            ]
        )

        for it in range(1, int(args.n_steps) + 1):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(seq_len), nu=2.5, length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=args.seed,
                alpha=1e-8,
            )
            gp.fit(x_obs, y_obs)

            best_idx = int(np.argmax(y_obs))
            best_y = float(y_obs[best_idx])
            x_best = x_obs[best_idx].copy()
            z_obs_mu = encode_batch_mu(vae, x_obs, device)
            z_best_enc = z_obs_mu[best_idx].copy()

            z_pool, _ = build_latent_pool(z_obs_mu, y_obs, replay_buffer, args, rng, latent_dim)
            x_pool = decode_batch_z_to_x(vae, z_pool, device)
            acq_pool = gp_acq(gp, x_pool, best_y, args.weight, args.xi, args.kappa)
            pool_logits = acquisition_log_tilt(acq_pool, args, scale=args.tau_temp)
            pool_weights = stable_softmax(pool_logits)

            flow_loss_hist, flow_loss_last = train_flow_weighted(flow, z_pool, pool_weights, args, device)
            flow_loss_mean = finite_mean(flow_loss_hist, default=np.nan)
            flow_entropy = estimate_flow_entropy(flow, args.flow_entropy_samples, device)
            flow_loss_last_hist.append(float(flow_loss_last))
            flow_loss_mean_hist.append(float(flow_loss_mean))
            entropy_hist.append(float(flow_entropy))
            replay_size_hist.append(float(replay_buffer.shape[0]))

            flow_iter_ckpt = os.path.join(flow_dir, f"flow_iter_{it:03d}.pt")
            save_flow_checkpoint(flow_iter_ckpt, flow, args, latent_dim, flow_loss_last_hist)
            plot_flow_training_overview(flow_overview_path, flow_loss_last_hist, flow_loss_mean_hist, entropy_hist)

            z0 = choose_chain_init(args, z_best_enc, replay_buffer, latent_dim, rng)
            z_cand, x_cand, z_chain, _x_chain, chain_logpi, accepted_states, mh_stats = mh_sample_latent_candidates(
                z0,
                flow,
                vae,
                gp,
                best_y,
                args,
                rng,
                device,
            )

            cand_util = np.asarray(gp_acq(gp, x_cand, best_y, args.select_acq, args.xi, args.kappa), dtype=np.float64)
            util_safe = np.where(np.isfinite(cand_util), cand_util, -np.inf)
            if np.all(~np.isfinite(util_safe)):
                util_safe = np.zeros_like(cand_util, dtype=np.float64)
            j = int(np.argmax(util_safe))

            z_next = z_cand[j]
            x_next = x_cand[j]
            y_next = float(oracle_f(x_next, target, step_size, w_close, w_smooth))

            z_obs = np.vstack([z_obs, z_next.reshape(1, -1)])
            x_obs = np.vstack([x_obs, x_next.reshape(1, -1)])
            y_obs = np.append(y_obs, y_next)
            replay_buffer = append_replay_buffer(replay_buffer, accepted_states, args.replay_max)

            best_so_far.append(float(np.max(y_obs)))
            acc_hist.append(float(mh_stats["accept_rate"]))

            wr.writerow(
                [
                    it,
                    y_next,
                    float(np.max(y_obs)),
                    mh_stats["accept_rate"],
                    float(np.max(util_safe)),
                    finite_mean(util_safe, default=np.nan),
                    float(chain_logpi[-1]) if chain_logpi.size else np.nan,
                    mh_stats["jump_norm_mean"],
                    mh_stats["log_alpha_mean"],
                    mh_stats["log_alpha_std"],
                    float(flow_loss_last),
                    float(flow_loss_mean),
                    float(flow_entropy),
                    mh_stats["frac_flow_proposals_used"],
                    mh_stats["frac_local_proposals_used"],
                    mh_stats["latent_norm_mean"],
                    int(z_pool.shape[0]),
                    int(replay_buffer.shape[0]),
                ]
            )
            f_csv.flush()

            if it == 1 or it % 5 == 0:
                print(
                    f"[COWBOYS-FLOW] it={it:03d} | y_next={y_next: .4f} | best={float(np.max(y_obs)): .4f} "
                    f"| acc={mh_stats['accept_rate']:.3f} | flow_loss={flow_loss_last:.4f}"
                )

            if mh_stats["accept_rate"] < 0.02:
                low_acc_count += 1
            else:
                low_acc_count = 0
            if low_acc_count >= 3:
                print(
                    "[WARN] MH acceptance is very low for several iterations. "
                    "Try lowering --beta_tilt, increasing --lambda_local, or increasing --sigma_local."
                )

            z_plot = np.vstack([z_bg, z_obs, z_cand]) if z_cand.shape[0] > 0 else np.vstack([z_bg, z_obs])
            zlim = resolve_latent_plot_limits(args, z_plot)
            gx, gy, a_grid, a_name = latent_grid_acq_2d(vae, gp, best_y, args, zlim, device)

            best_idx_now = int(np.argmax(y_obs))
            z_best_now = z_obs[best_idx_now].copy()

            plot_step(
                save_path=os.path.join(step_dir, f"step_{it:03d}.png"),
                z_bg=z_bg,
                z_obs=z_obs,
                y_obs=y_obs,
                z_next=z_next,
                z_best=z_best_now,
                z_cand=z_cand,
                x_next=x_next,
                y_next=y_next,
                step_size=step_size,
                best_so_far=best_so_far,
                zlim=zlim,
                gx=gx,
                gy=gy,
                a_grid=a_grid,
                a_name=a_name,
            )

            z_flow_diag = sample_flow_np(flow, args.n_flow_diag, device)
            z_prior_diag = rng.standard_normal(size=(int(args.n_flow_diag), latent_dim))
            z_replay_diag = replay_buffer[-min(replay_buffer.shape[0], int(args.n_flow_diag)) :] if replay_buffer.shape[0] else replay_buffer
            plot_latent_flow_diag(
                save_path=os.path.join(diag_dir, f"latent_flow_iter_{it:03d}.png"),
                z_bg=z_bg,
                z_pool=z_pool,
                pool_weights=pool_weights,
                z_flow=z_flow_diag,
                z_prior=z_prior_diag,
                z_replay=z_replay_diag,
                z_chain=z_chain,
                iteration=it,
            )

    best_idx = int(np.argmax(y_obs))
    z_best = z_obs[best_idx]
    x_best = x_obs[best_idx]
    y_best = float(y_obs[best_idx])
    zlim_final = resolve_latent_plot_limits(args, np.vstack([z_bg, z_obs]))

    final_png = os.path.join(plot_root, "final_summary_cowboys_flow_latent.png")
    plot_final(
        save_path=final_png,
        z_bg=z_bg,
        z_obs=z_obs,
        y_obs=y_obs,
        z_best=z_best,
        x_best=x_best,
        y_best=y_best,
        step_size=step_size,
        best_so_far=best_so_far,
        acc_hist=acc_hist,
        zlim=zlim_final,
    )

    final_flow_ckpt = os.path.join(flow_dir, "flow_final.pt")
    save_flow_checkpoint(final_flow_ckpt, flow, args, latent_dim, flow_loss_last_hist)

    trace_path = os.path.join(plot_root, "trace_cowboys_flow_latent.npz")
    np.savez_compressed(
        fs_path(trace_path),
        z_obs=z_obs,
        x_obs=x_obs,
        y_obs=y_obs,
        best_so_far=np.asarray(best_so_far, dtype=np.float64),
        acc_hist=np.asarray(acc_hist, dtype=np.float64),
        flow_loss_last_hist=np.asarray(flow_loss_last_hist, dtype=np.float64),
        flow_loss_mean_hist=np.asarray(flow_loss_mean_hist, dtype=np.float64),
        entropy_hist=np.asarray(entropy_hist, dtype=np.float64),
        replay_size_hist=np.asarray(replay_size_hist, dtype=np.float64),
    )

    print("\n=== COWBOYS latent-flow finished ===")
    print("Best oracle:", y_best)
    print("Saved steps:", os.path.abspath(step_dir))
    print("Saved diagnostics:", os.path.abspath(diag_dir))
    print("Saved flow checkpoints:", os.path.abspath(flow_dir))
    print("Saved final summary:", os.path.abspath(final_png))
    print("Saved metrics:", os.path.abspath(metrics_path))
    print("Saved trace:", os.path.abspath(trace_path))


if __name__ == "__main__":
    main()
