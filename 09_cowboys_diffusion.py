import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from toy_common import (
    ensure_dir,
    expected_improvement,
    load_config,
    norm_cdf,
    oracle_f,
    set_seed,
    turtle_path,
)

# NOTE:
# This script uses a diffusion model as a proposal mechanism in MH.
# The default target is acquisition-tilted (plus optional Gaussian prior),
# not "exact diffusion prior times acquisition".


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
    def __init__(self, x_dim: int, time_dim: int = 64, hidden: int = 512, depth: int = 5):
        super().__init__()
        self.temb = SinusoidalTimeEmb(time_dim)
        d = x_dim + time_dim
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor, t: torch.Tensor):
        return self.net(torch.cat([xt, self.temb(t)], dim=1))


def make_beta_schedule(T: int, beta_start: float, beta_end: float):
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)


def ddpm_arrays_np(betas: np.ndarray):
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    beta_tilde = np.zeros_like(betas)
    beta_tilde[0] = betas[0]
    for t in range(1, len(betas)):
        beta_tilde[t] = betas[t] * (1.0 - abar[t - 1]) / max(1e-12, 1.0 - abar[t])
    return alphas, abar, beta_tilde


def build_time_indices(T: int, S: int, mode: str, high_window_frac: float):
    """
    Select diffusion timesteps used in the augmented forward/reverse path.

    For exact one-step DDPM kernel bookkeeping in this implementation, use only
    contiguous schedules.
    """
    S_eff = max(1, min(int(S), int(T) - 1))
    if mode == "low":
        return np.arange(1, S_eff + 1, dtype=np.int64)
    if mode == "high_window":
        if high_window_frac > 0.0:
            S_eff = max(1, min(S_eff, int(np.ceil(float(high_window_frac) * (T - 1)))))
        start = max(1, T - S_eff)
        return np.arange(start, T, dtype=np.int64)
    raise ValueError("For exact MH in this codepath, mh_time_mode must be one of: low, high_window")


def train_diffusion(model, X_scaled, betas, device, steps, batch, lr, wd, grad_clip, log_every):
    model.train()
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    N = X_t.shape[0]
    T = len(betas)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    abar_t = torch.cumprod(1.0 - betas_t, dim=0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        x0 = X_t[idx]
        t = torch.randint(0, T, (batch,), device=device)
        ab = abar_t[t].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
        eps_hat = model(xt, t)
        loss = torch.mean((eps_hat - eps) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if step == 1 or step % max(1, log_every) == 0:
            print(f"[DIFF-TRAIN] step={step:6d}/{steps} loss={losses[-1]:.6f}")
    model.eval()
    return np.asarray(losses, dtype=np.float64)


def save_diffusion(path, model, args, losses, x_dim):
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "state_dict": model.state_dict(),
        "x_dim": int(x_dim),
        "T": int(args.diff_T),
        "beta_start": float(args.diff_beta_start),
        "beta_end": float(args.diff_beta_end),
        "time_dim": int(args.diff_time_dim),
        "hidden": int(args.diff_hidden),
        "depth": int(args.diff_depth),
        "loss_hist": losses.astype(np.float32),
    }
    torch.save(ckpt, path)


def torch_load_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_diffusion(path, device):
    ckpt = torch_load_compat(path, device)
    x_dim = int(ckpt["x_dim"])
    model = EpsMLP(
        x_dim=x_dim,
        time_dim=int(ckpt.get("time_dim", 64)),
        hidden=int(ckpt.get("hidden", 512)),
        depth=int(ckpt.get("depth", 5)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas = make_beta_schedule(
        int(ckpt["T"]),
        float(ckpt["beta_start"]),
        float(ckpt["beta_end"]),
    )
    return model, betas, x_dim, np.asarray(ckpt.get("loss_hist", []), dtype=np.float64)


@torch.no_grad()
def reverse_mean_np(model, x_t_np, t_idx, alphas, abar, betas, device):
    xt = torch.tensor(x_t_np, dtype=torch.float32, device=device).view(1, -1)
    tt = torch.tensor([int(t_idx)], dtype=torch.long, device=device)
    eps_hat = model(xt, tt)
    alpha_t = float(alphas[t_idx])
    abar_t = float(abar[t_idx])
    beta_t = float(betas[t_idx])
    mean = (1.0 / np.sqrt(alpha_t)) * (
        xt - (beta_t / np.sqrt(max(1e-12, 1.0 - abar_t))) * eps_hat
    )
    return mean.detach().cpu().numpy().reshape(-1).astype(np.float64)


@torch.no_grad()
def sample_prior(model, betas, n_samples, device):
    x_dim = model.net[-1].out_features
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas_t = 1.0 - betas_t
    abar_t = torch.cumprod(alphas_t, dim=0)
    x = torch.randn(n_samples, x_dim, device=device)
    for t in reversed(range(len(betas))):
        tt = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps_hat = model(x, tt)
        alpha = alphas_t[t]
        ab = abar_t[t]
        beta = betas_t[t]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(torch.clamp(1.0 - ab, min=1e-12))) * eps_hat)
        if t > 0:
            ab_prev = abar_t[t - 1]
            beta_tilde = beta * (1.0 - ab_prev) / torch.clamp(1.0 - ab, min=1e-12)
            sigma = torch.sqrt(torch.clamp(beta_tilde, min=1e-20))
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x.detach().cpu().numpy().astype(np.float64)


def gp_acq(gp, X, best_y, acq, xi, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    if acq == "ei":
        return expected_improvement(mu, sigma, best_y=best_y, xi=xi)
    if acq == "ucb":
        return mu + kappa * sigma
    if acq == "pi":
        return norm_cdf((mu - best_y - xi) / sigma)
    raise ValueError("acq must be one of: ei, ucb, pi")


def log_normal_iso(x, mean, var):
    d = x.shape[0]
    diff = x - mean
    return -0.5 * (d * np.log(2.0 * np.pi * var) + float(np.dot(diff, diff) / max(var, 1e-20)))


def log_pi_target(x_scaled, gp, best_y, args):
    if args.state_space == "box":
        if np.any(x_scaled < -1.0) or np.any(x_scaled > 1.0):
            return -np.inf, np.nan

    x_rad = x_scaled.reshape(1, -1) * np.pi
    a = float(gp_acq(gp, x_rad, best_y, args.weight, args.xi, args.kappa)[0])

    if args.tilt_form == "exp":
        logv = args.beta_tilt * a
    elif args.tilt_form == "power":
        if args.weight not in {"ei", "pi"}:
            raise ValueError("tilt_form='power' is only defined for weight in {'ei','pi'}.")
        logv = args.beta_tilt * np.log(max(a, 1e-12))
    else:
        raise ValueError("tilt_form must be one of: exp, power")

    if args.target_prior == "gaussian":
        var = args.prior_sigma * args.prior_sigma
        d = x_scaled.shape[0]
        logv += -0.5 * (d * np.log(2.0 * np.pi * var) + float(np.dot(x_scaled, x_scaled) / max(var, 1e-20)))
    return float(logv), float(a)


def mh_diffusion_step(
    x_current,
    model,
    gp,
    best_y,
    time_indices,
    betas,
    alphas,
    abar,
    beta_tilde,
    args,
    rng,
    device,
):
    S = int(len(time_indices))
    d = x_current.shape[0]

    x_fwd = [None] * (S + 1)
    x_fwd[0] = x_current.copy()
    log_fwd = 0.0
    for s in range(1, S + 1):
        t_idx = int(time_indices[s - 1])
        mean_q = np.sqrt(alphas[t_idx]) * x_fwd[s - 1]
        var_q = float(
            max(
                float(betas[t_idx]) * float(args.mh_var_scale),
                float(args.mh_var_floor),
            )
        )
        x_fwd[s] = mean_q + np.sqrt(var_q) * rng.standard_normal(d)
        log_fwd += log_normal_iso(x_fwd[s], mean_q, var_q)

    x_rev = [None] * (S + 1)
    x_rev[S] = x_fwd[S].copy()
    log_rev = 0.0
    for s in range(S, 0, -1):
        t_idx = int(time_indices[s - 1])
        mean_r = reverse_mean_np(model, x_rev[s], t_idx, alphas, abar, betas, device)
        var_r = float(
            max(
                float(beta_tilde[t_idx]) * float(args.mh_var_scale),
                float(args.mh_var_floor),
            )
        )
        x_rev[s - 1] = mean_r + np.sqrt(var_r) * rng.standard_normal(d)
        log_rev += log_normal_iso(x_rev[s - 1], mean_r, var_r)
    x_prop = x_rev[0].copy()

    log_fwd_sw = 0.0
    for s in range(1, S + 1):
        t_idx = int(time_indices[s - 1])
        mean_q_sw = np.sqrt(alphas[t_idx]) * x_rev[s - 1]
        var_q_sw = float(
            max(
                float(betas[t_idx]) * float(args.mh_var_scale),
                float(args.mh_var_floor),
            )
        )
        log_fwd_sw += log_normal_iso(x_rev[s], mean_q_sw, var_q_sw)

    log_rev_sw = 0.0
    for s in range(1, S + 1):
        t_idx = int(time_indices[s - 1])
        mean_r_sw = reverse_mean_np(model, x_fwd[s], t_idx, alphas, abar, betas, device)
        var_r_sw = float(
            max(
                float(beta_tilde[t_idx]) * float(args.mh_var_scale),
                float(args.mh_var_floor),
            )
        )
        log_rev_sw += log_normal_iso(x_fwd[s - 1], mean_r_sw, var_r_sw)

    log_pi_prop, _ = log_pi_target(x_prop, gp, best_y, args)
    log_pi_cur, _ = log_pi_target(x_current, gp, best_y, args)

    if not np.isfinite(log_pi_cur):
        # Defensive fallback: keep state unchanged if the current point has invalid target density.
        return x_current.copy(), {"accepted": False, "log_alpha": -np.inf, "jump_norm": 0.0}

    if not np.isfinite(log_pi_prop):
        log_alpha = -np.inf
    else:
        ratio = (log_pi_prop + log_fwd_sw + log_rev_sw) - (log_pi_cur + log_fwd + log_rev)
        log_alpha = min(0.0, float(ratio))

    accepted = bool(np.log(rng.random()) < log_alpha)
    x_next = x_prop if accepted else x_current.copy()
    info = {
        "accepted": accepted,
        "log_alpha": float(log_alpha),
        "jump_norm": float(np.linalg.norm(x_prop - x_current)),
    }
    return x_next, info


def mh_sample_candidates(
    x0_scaled,
    model,
    gp,
    best_y,
    time_indices,
    betas,
    alphas,
    abar,
    beta_tilde,
    args,
    rng,
    device,
):
    x = x0_scaled.copy()
    accepted = 0
    samples = []
    chain = []
    chain_logpi = []
    log_alpha_trace = []
    jump_norms = []

    for step in range(1, args.mh_steps + 1):
        x, info = mh_diffusion_step(
            x,
            model,
            gp,
            best_y,
            time_indices,
            betas,
            alphas,
            abar,
            beta_tilde,
            args,
            rng,
            device,
        )
        accepted += int(info["accepted"])
        log_alpha_trace.append(info["log_alpha"])
        jump_norms.append(info["jump_norm"])

        if step % args.mh_thin == 0:
            chain.append(x.copy())
            logpi_x, _ = log_pi_target(x, gp, best_y, args)
            chain_logpi.append(logpi_x)
        if step > args.mh_burn and ((step - args.mh_burn) % args.mh_thin == 0):
            samples.append(x.copy())

    d = x0_scaled.shape[0]
    samples = np.array(samples, dtype=np.float64) if samples else np.zeros((0, d), dtype=np.float64)
    chain = np.array(chain, dtype=np.float64) if chain else np.zeros((0, d), dtype=np.float64)
    chain_logpi = np.array(chain_logpi, dtype=np.float64) if chain_logpi else np.zeros((0,), dtype=np.float64)
    stats = {
        "accept_rate": float(accepted / max(1, args.mh_steps)),
        "log_alpha_mean": float(np.mean(log_alpha_trace)) if log_alpha_trace else 0.0,
        "log_alpha_std": float(np.std(log_alpha_trace)) if log_alpha_trace else 0.0,
        "jump_norm_mean": float(np.mean(jump_norms)) if jump_norms else 0.0,
    }
    return samples, chain, chain_logpi, stats


def plot_diff_loss(losses, path):
    if losses is None or len(losses) == 0:
        return
    plt.figure(figsize=(7, 4.6))
    plt.plot(losses, linewidth=1.2)
    plt.title("Sequence diffusion training loss")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_prior_diag(X_data, X_prior, pca, path):
    Zd = pca.transform(X_data)
    Zp = pca.transform(X_prior)
    plt.figure(figsize=(7, 5.8))
    plt.scatter(Zd[:, 0], Zd[:, 1], s=8, alpha=0.15, linewidth=0.0, label="dataset")
    plt.scatter(Zp[:, 0], Zp[:, 1], s=9, alpha=0.25, linewidth=0.0, label="diffusion prior")
    plt.title("Prior samples vs dataset (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def pca_grid_acq(pca, gp, best_y, args, xlim, ylim):
    z1 = np.linspace(xlim[0], xlim[1], args.grid_res)
    z2 = np.linspace(ylim[0], ylim[1], args.grid_res)
    gx, gy = np.meshgrid(z1, z2)
    Zg = np.stack([gx.ravel(), gy.ravel()], axis=1)
    Xg = pca.inverse_transform(Zg)
    A = gp_acq(gp, Xg, best_y, args.plot_acq, args.xi, args.kappa).reshape(args.grid_res, args.grid_res)
    name = {"ei": "EI(x)", "ucb": "UCB(x)", "pi": "PI(x)"}[args.plot_acq]
    return gx, gy, A, name


def plot_step(
    save_path,
    X_bg,
    X_obs,
    y_obs,
    x_next,
    y_next,
    x_best,
    y_best,
    X_chain,
    X_cand,
    cand_util,
    selected_util,
    target_pts,
    step_size,
    best_so_far,
    acc_hist,
    chain_logpi,
    pca,
    gx,
    gy,
    A_grid,
    A_name,
):
    fig = plt.figure(figsize=(18.0, 5.0))

    Z_bg = pca.transform(X_bg)
    Z_obs = pca.transform(X_obs)
    Z_next = pca.transform(x_next.reshape(1, -1))[0]
    Z_best = pca.transform(x_best.reshape(1, -1))[0]
    Z_chain = pca.transform(X_chain) if X_chain.shape[0] else np.zeros((0, 2))
    Z_cand = pca.transform(X_cand) if X_cand.shape[0] else np.zeros((0, 2))

    ax1 = fig.add_subplot(1, 4, 1)
    im = ax1.contourf(gx, gy, A_grid, levels=35)
    plt.colorbar(im, ax=ax1, label=A_name)
    ax1.scatter(Z_bg[:, 0], Z_bg[:, 1], s=8, alpha=0.10, linewidth=0.0, label="data")
    ax1.scatter(Z_obs[:, 0], Z_obs[:, 1], c=y_obs, s=52, edgecolor="black", linewidth=0.35, label="evaluated")
    if Z_chain.shape[0] > 1:
        ax1.plot(Z_chain[:, 0], Z_chain[:, 1], linewidth=1.0, alpha=0.9, label="MH chain")
    if Z_cand.shape[0] > 0:
        ax1.scatter(Z_cand[:, 0], Z_cand[:, 1], s=22, alpha=0.75, marker="x", label="candidates")
    ax1.scatter([Z_best[0]], [Z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0, label="best")
    ax1.scatter([Z_next[0]], [Z_next[1]], c="yellow", s=150, marker="D", edgecolor="black", linewidth=1.0, label="next")
    ax1.set_title("PCA acquisition + MH proposal")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 4, 2)
    pts_target = target_pts
    pts_next = turtle_path(x_next, step_size)
    ax2.plot(pts_target[:, 0], pts_target[:, 1], linestyle="--", linewidth=2.0, alpha=0.8, label="target")
    ax2.plot(pts_next[:, 0], pts_next[:, 1], linewidth=2.4, label="x_next")
    ax2.scatter([pts_next[0, 0]], [pts_next[0, 1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Decoded x_next\ny={y_next:.3f}")
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(best_so_far, marker="o", linewidth=1.4, label="best-so-far y")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("best oracle")
    ax3_t = ax3.twinx()
    ax3_t.plot(acc_hist, color="tab:orange", marker="x", linewidth=1.3, label="MH accept")
    ax3_t.set_ylabel("accept rate")
    ax3.set_title("Progress + acceptance")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_t.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="best")

    ax4 = fig.add_subplot(1, 4, 4)
    if cand_util.size > 0:
        util = np.asarray(cand_util, dtype=np.float64)
        util = util[np.isfinite(util)]
        if util.size > 0:
            umin = float(np.min(util))
            umax = float(np.max(util))
            tiny = max(1e-12, 1e-9 * max(1.0, abs(umin), abs(umax)))
            if abs(umax - umin) <= tiny:
                center = 0.5 * (umin + umax)
                half = max(1e-3, 1e-2 * max(1.0, abs(center)))
                bins = np.linspace(center - half, center + half, 11)
            else:
                n_bins = min(25, max(5, util.size // 5))
                bins = np.linspace(umin, umax, n_bins + 1)
            ax4.hist(util, bins=bins, alpha=0.8)
        if np.isfinite(selected_util):
            ax4.axvline(selected_util, color="red", linestyle="--", linewidth=2.0, label="selected util")
    if chain_logpi.size > 0:
        ax4_t = ax4.twinx()
        ax4_t.plot(chain_logpi, color="tab:green", linewidth=1.2, alpha=0.8, label="chain log pi")
        h3, l3 = ax4.get_legend_handles_labels()
        h4, l4 = ax4_t.get_legend_handles_labels()
        ax4.legend(h3 + h4, l3 + l4, loc="best")
    ax4.set_title("Candidate utility + chain energy")
    ax4.set_xlabel("utility")

    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close(fig)


def plot_final(
    save_path,
    X_bg,
    X_obs,
    y_obs,
    x_best,
    y_best,
    target_pts,
    step_size,
    best_so_far,
    acc_hist,
    pca,
):
    fig = plt.figure(figsize=(16.0, 5.0))
    Z_bg = pca.transform(X_bg)
    Z_obs = pca.transform(X_obs)
    Z_best = pca.transform(x_best.reshape(1, -1))[0]

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(Z_bg[:, 0], Z_bg[:, 1], s=9, alpha=0.12, linewidth=0.0, label="data")
    sc = ax1.scatter(Z_obs[:, 0], Z_obs[:, 1], c=y_obs, s=52, edgecolor="black", linewidth=0.35, label="BO points")
    plt.colorbar(sc, ax=ax1, label="oracle y")
    ax1.scatter([Z_best[0]], [Z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0, label="best")
    ax1.set_title("Final PCA overlay")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 4, 2)
    pts_target = target_pts
    pts_best = turtle_path(x_best, step_size)
    ax2.plot(pts_target[:, 0], pts_target[:, 1], linestyle="--", linewidth=2.0, alpha=0.8, label="target")
    ax2.plot(pts_best[:, 0], pts_best[:, 1], linewidth=2.4, label="best")
    ax2.scatter([pts_best[0, 0]], [pts_best[0, 1]], s=70)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    ax2.set_title(f"Best decoded path\ny={y_best:.3f}")
    ax2.legend(loc="best")

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
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--plotroot", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--n_steps", type=int, default=40)

    ap.add_argument("--weight", type=str, default="ei", choices=["pi", "ei", "ucb"])
    ap.add_argument("--beta_tilt", type=float, default=8.0)
    ap.add_argument("--tilt_form", type=str, default="exp", choices=["exp", "power"])
    ap.add_argument("--select_acq", type=str, default="ei", choices=["pi", "ei", "ucb"])
    ap.add_argument("--plot_acq", type=str, default="ei", choices=["pi", "ei", "ucb"])
    ap.add_argument(
        "--allow_mismatched_select",
        action="store_true",
        help="Allow select_acq != weight. By default they are forced to match for target/proposal consistency.",
    )
    ap.add_argument("--xi", type=float, default=0.06)
    ap.add_argument("--kappa", type=float, default=2.2)
    ap.add_argument("--target_prior", type=str, default="none", choices=["none", "gaussian"])
    ap.add_argument("--prior_sigma", type=float, default=1.0)
    ap.add_argument(
        "--state_space",
        type=str,
        default="box",
        choices=["box", "unbounded"],
        help="State-space support for MH target. 'box' enforces [-1,1]^L support via log_pi=-inf outside.",
    )

    ap.add_argument("--mh_steps", type=int, default=1200)
    ap.add_argument("--mh_burn", type=int, default=300)
    ap.add_argument("--mh_thin", type=int, default=4)
    ap.add_argument("--mh_S", type=int, default=12)
    ap.add_argument(
        "--mh_time_mode",
        type=str,
        default="low",
        choices=["low", "high_window"],
        help="Exact schedules in this implementation: contiguous low steps or contiguous late-window steps.",
    )
    ap.add_argument(
        "--mh_high_window_frac",
        type=float,
        default=0.35,
        help="If mh_time_mode=high_window, fraction of late timesteps used.",
    )
    ap.add_argument("--mh_var_scale", type=float, default=10.0, help="Multiply q/r proposal variances by this factor.")
    ap.add_argument("--mh_var_floor", type=float, default=1e-3, help="Lower bound for q/r proposal variances.")
    ap.add_argument("--chain_init", type=str, default="best", choices=["best", "random"])

    ap.add_argument("--diff_ckpt", type=str, default=None)
    ap.add_argument("--force_retrain_diffusion", action="store_true")
    ap.add_argument("--diff_T", type=int, default=200)
    ap.add_argument("--diff_beta_start", type=float, default=1e-4)
    ap.add_argument("--diff_beta_end", type=float, default=2e-2)
    ap.add_argument("--diff_time_dim", type=int, default=64)
    ap.add_argument("--diff_hidden", type=int, default=512)
    ap.add_argument("--diff_depth", type=int, default=5)
    ap.add_argument("--diff_train_steps", type=int, default=12000)
    ap.add_argument("--diff_batch", type=int, default=512)
    ap.add_argument("--diff_lr", type=float, default=2e-4)
    ap.add_argument("--diff_weight_decay", type=float, default=1e-4)
    ap.add_argument("--diff_grad_clip", type=float, default=1.0)
    ap.add_argument("--diff_log_every", type=int, default=250)

    ap.add_argument("--grid_res", type=int, default=110)
    ap.add_argument("--n_data_scatter", type=int, default=6000)
    ap.add_argument("--n_prior_diag", type=int, default=2500)
    args = ap.parse_args()

    if (not args.allow_mismatched_select) and (args.select_acq != args.weight):
        print(
            f"[INFO] forcing select_acq to match weight: {args.select_acq} -> {args.weight}. "
            "Use --allow_mismatched_select to disable this."
        )
        args.select_acq = args.weight

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_npz = os.path.join(args.outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}"
    data = np.load(data_npz, allow_pickle=True)
    X_all = data["X"].astype(np.float64)
    target = data["target"].astype(np.float64)
    step_size, w_close, w_smooth = load_config(args.outdir)

    N, L = X_all.shape
    X_all_scaled = (X_all / np.pi).astype(np.float32)
    print(f"Loaded dataset: N={N}, L={L}")
    print(
        f"[TARGET] acquisition-tilted MH target | weight={args.weight} "
        f"| tilt_form={args.tilt_form} | beta_tilt={args.beta_tilt} "
        f"| prior={args.target_prior} | state_space={args.state_space}"
    )
    print(
        f"[PROPOSAL] DDPM one-step path kernels | mh_time_mode={args.mh_time_mode} "
        f"(contiguous schedule only for exactness in this implementation)"
    )

    if args.plotroot is None:
        args.plotroot = os.path.join(
            args.outdir, "09_cowboys_diffusion", f"{args.weight}_bt{args.beta_tilt}_mh{args.mh_steps}_S{args.mh_S}"
        )
    plot_root = args.plotroot
    step_dir = os.path.join(plot_root, "steps")
    diag_dir = os.path.join(plot_root, "diagnostics")
    ensure_dir(step_dir)
    ensure_dir(diag_dir)

    diff_ckpt = args.diff_ckpt or os.path.join(args.outdir, "models", "sequence_diffusion.pt")
    retrain = args.force_retrain_diffusion or (not os.path.exists(diff_ckpt))
    if retrain:
        print("[DIFF] training sequence diffusion model...")
        model = EpsMLP(
            x_dim=L, time_dim=args.diff_time_dim, hidden=args.diff_hidden, depth=args.diff_depth
        ).to(device)
        betas = make_beta_schedule(args.diff_T, args.diff_beta_start, args.diff_beta_end)
        losses = train_diffusion(
            model,
            X_all_scaled,
            betas,
            device,
            args.diff_train_steps,
            args.diff_batch,
            args.diff_lr,
            args.diff_weight_decay,
            args.diff_grad_clip,
            args.diff_log_every,
        )
        save_diffusion(diff_ckpt, model, args, losses, L)
        print("[DIFF] saved:", os.path.abspath(diff_ckpt))
    else:
        print("[DIFF] loading existing checkpoint:", diff_ckpt)

    model, betas, x_dim, losses_ckpt = load_diffusion(diff_ckpt, device)
    assert x_dim == L, f"x_dim mismatch: {x_dim} vs {L}"
    print(f"[DIFF] loaded | T={len(betas)} | x_dim={x_dim}")
    plot_diff_loss(losses_ckpt, os.path.join(diag_dir, "diffusion_train_loss.png"))

    prior_scaled = sample_prior(model, betas, args.n_prior_diag, device)
    if args.state_space == "box":
        prior_scaled = np.clip(prior_scaled, -1.0, 1.0)
    prior_rad = prior_scaled * np.pi

    n_bg = min(args.n_data_scatter, N)
    idx_bg = rng.choice(N, size=n_bg, replace=False)
    X_bg = X_all[idx_bg]
    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(X_bg)
    plot_prior_diag(X_bg, prior_rad[:n_bg], pca, os.path.join(diag_dir, "prior_vs_data_pca.png"))

    prior_mean_mse = float(np.mean((X_all_scaled[idx_bg].mean(axis=0) - prior_scaled[:n_bg].mean(axis=0)) ** 2))
    prior_std_mse = float(np.mean((X_all_scaled[idx_bg].std(axis=0) - prior_scaled[:n_bg].std(axis=0)) ** 2))
    print(f"[DIFF-DIAG] mean_mse={prior_mean_mse:.6f} | std_mse={prior_std_mse:.6f}")

    # Initial BO design from diffusion prior
    x_init_scaled = sample_prior(model, betas, args.n_init, device)
    if args.state_space == "box":
        x_init_scaled = np.clip(x_init_scaled, -1.0, 1.0)
    X_obs = x_init_scaled * np.pi
    y_obs = np.array(
        [float(oracle_f(x, target, step_size, w_close, w_smooth)) for x in X_obs],
        dtype=np.float64,
    )

    print("[INIT] best y:", float(y_obs.max()))
    best_so_far = [float(y_obs.max())]
    acc_hist = [np.nan]
    low_acc_count = 0

    alphas, abar, beta_tilde = ddpm_arrays_np(betas)
    time_indices = build_time_indices(
        T=len(betas),
        S=args.mh_S,
        mode=args.mh_time_mode,
        high_window_frac=args.mh_high_window_frac,
    )
    print(
        f"[MH] using S={len(time_indices)} | mode={args.mh_time_mode} "
        f"| t_min={int(time_indices.min())} t_max={int(time_indices.max())}"
    )

    metrics_path = os.path.join(plot_root, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f_csv:
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
            ]
        )

        for it in range(1, args.n_steps + 1):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(L), nu=2.5, length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=args.seed,
                alpha=1e-8,
            )
            gp.fit(X_obs, y_obs)

            best_idx = int(np.argmax(y_obs))
            best_y = float(y_obs[best_idx])
            x_best = X_obs[best_idx].copy()

            if args.chain_init == "best":
                x0_chain_scaled = x_best / np.pi
                if args.state_space == "box":
                    x0_chain_scaled = np.clip(x0_chain_scaled, -1.0, 1.0)
            else:
                x0_chain_scaled = sample_prior(model, betas, 1, device)[0]
                if args.state_space == "box":
                    x0_chain_scaled = np.clip(x0_chain_scaled, -1.0, 1.0)

            cand_scaled, chain_scaled, chain_logpi, mh_stats = mh_sample_candidates(
                x0_chain_scaled,
                model,
                gp,
                best_y,
                time_indices,
                betas,
                alphas,
                abar,
                beta_tilde,
                args,
                rng,
                device,
            )
            fallback_used = False
            if cand_scaled.shape[0] == 0:
                fallback_used = True
                cand_scaled = sample_prior(model, betas, max(16, args.mh_thin * 2), device)
            # Keep accepted MH states untouched. Only bound direct prior/fallback draws.
            if args.state_space == "box" and fallback_used:
                cand_scaled = np.clip(cand_scaled, -1.0, 1.0)

            X_cand = cand_scaled * np.pi
            X_chain = chain_scaled * np.pi if chain_scaled.shape[0] else np.zeros((0, L), dtype=np.float64)
            cand_util = np.asarray(gp_acq(gp, X_cand, best_y, args.select_acq, args.xi, args.kappa), dtype=np.float64)
            util_safe = np.where(np.isfinite(cand_util), cand_util, -np.inf)
            if np.all(~np.isfinite(util_safe)):
                util_safe = np.zeros_like(cand_util, dtype=np.float64)
            j = int(np.argmax(util_safe))
            x_next = X_cand[j]
            y_next = float(oracle_f(x_next, target, step_size, w_close, w_smooth))

            X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
            y_obs = np.append(y_obs, y_next)
            best_so_far.append(float(y_obs.max()))
            acc_hist.append(float(mh_stats["accept_rate"]))

            wr.writerow(
                [
                    it,
                    y_next,
                    float(y_obs.max()),
                    mh_stats["accept_rate"],
                    float(np.max(util_safe)),
                    float(np.mean(util_safe)),
                    float(chain_logpi[-1]) if chain_logpi.size else np.nan,
                    mh_stats["jump_norm_mean"],
                    mh_stats["log_alpha_mean"],
                    mh_stats["log_alpha_std"],
                ]
            )
            f_csv.flush()

            if it == 1 or it % 5 == 0:
                print(
                    f"[COWBOYS-DIFF] it={it:03d} | y_next={y_next: .4f} | best={float(y_obs.max()): .4f} "
                    f"| acc={mh_stats['accept_rate']:.3f} | cand_{args.select_acq}_max={float(np.max(util_safe)):.4g}"
                )
            if mh_stats["accept_rate"] < 0.02:
                low_acc_count += 1
            else:
                low_acc_count = 0
            if low_acc_count >= 3:
                print(
                    "[WARN] MH acceptance is very low for several iterations. "
                    "Try lowering --beta_tilt, reducing --mh_S, or increasing --mh_var_scale/--mh_var_floor."
                )

            Z_bg = pca.transform(X_bg)
            qx = np.quantile(Z_bg[:, 0], [0.02, 0.98])
            qy = np.quantile(Z_bg[:, 1], [0.02, 0.98])
            sx = float(qx[1] - qx[0])
            sy = float(qy[1] - qy[0])
            xlim = (float(qx[0] - 0.15 * sx), float(qx[1] + 0.15 * sx))
            ylim = (float(qy[0] - 0.15 * sy), float(qy[1] + 0.15 * sy))
            gx, gy, A_grid, A_name = pca_grid_acq(pca, gp, best_y, args, xlim, ylim)

            best_idx_now = int(np.argmax(y_obs))
            x_best_now = X_obs[best_idx_now]
            y_best_now = float(y_obs[best_idx_now])
            plot_step(
                save_path=os.path.join(step_dir, f"step_{it:03d}.png"),
                X_bg=X_bg,
                X_obs=X_obs,
                y_obs=y_obs,
                x_next=x_next,
                y_next=y_next,
                x_best=x_best_now,
                y_best=y_best_now,
                X_chain=X_chain,
                X_cand=X_cand,
                cand_util=util_safe,
                selected_util=float(util_safe[j]),
                target_pts=target,
                step_size=step_size,
                best_so_far=best_so_far,
                acc_hist=acc_hist,
                chain_logpi=chain_logpi,
                pca=pca,
                gx=gx,
                gy=gy,
                A_grid=A_grid,
                A_name=A_name,
            )

    best_idx = int(np.argmax(y_obs))
    x_best = X_obs[best_idx]
    y_best = float(y_obs[best_idx])
    final_png = os.path.join(plot_root, "final_summary_cowboys_diffusion.png")
    plot_final(
        save_path=final_png,
        X_bg=X_bg,
        X_obs=X_obs,
        y_obs=y_obs,
        x_best=x_best,
        y_best=y_best,
        target_pts=target,
        step_size=step_size,
        best_so_far=best_so_far,
        acc_hist=acc_hist,
        pca=pca,
    )

    trace_path = os.path.join(plot_root, "trace_cowboys_diffusion.npz")
    np.savez_compressed(
        trace_path,
        X_obs=X_obs,
        y_obs=y_obs,
        best_so_far=np.array(best_so_far, dtype=np.float64),
        acc_hist=np.array(acc_hist, dtype=np.float64),
        prior_mean_mse=np.array([prior_mean_mse], dtype=np.float64),
        prior_std_mse=np.array([prior_std_mse], dtype=np.float64),
    )
    print("\n=== COWBOYS diffusion (sequence-space) finished ===")
    print("Best oracle:", y_best)
    print("Saved steps:", os.path.abspath(step_dir))
    print("Saved diagnostics:", os.path.abspath(diag_dir))
    print("Saved final summary:", os.path.abspath(final_png))
    print("Saved metrics:", os.path.abspath(metrics_path))
    print("Saved trace:", os.path.abspath(trace_path))


if __name__ == "__main__":
    main()
