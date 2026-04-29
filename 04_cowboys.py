#python 04_cowboys2.py --outdir toy_circle_data --n_steps 60 --acq pi --kappa 2.2 --tau 0.5 --pcn_beta 0.35

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern

from toy_common import (
    decode_z_to_delta_theta,
    encode_dataset_to_z,
    ensure_dir,
    expected_improvement,
    load_vae,
    norm_cdf,
    oracle_f,
    set_seed,
    turtle_path,
)


# =========================
# Acquisition functions
# =========================
def ucb(mu, sigma, kappa=2.0):
    return mu + kappa * sigma

def probability_of_improvement(mu, sigma, best_y, xi=0.0):
    """
    PI for maximization: P(f > best_y + xi).
    mu, sigma can be arrays.
    """
    sigma = np.maximum(sigma, 1e-12)
    Z = (mu - best_y - xi) / sigma
    return norm_cdf(Z)


# =========================
# pCN MCMC sampler on latent space
# Target:
#   - For ei/ucb:  pi(z) ∝ exp(tau * acq(z)) N(0,I)
#   - For pi:      pi(z) ∝ [PI(z)]^tau      N(0,I)   (paper-like "likelihood tilt")
# pCN proposal:
#   z' = sqrt(1-beta^2) z + beta * N(0,I)
# With pCN, the N(0,I) prior cancels in the MH ratio.
# =========================
def pcn_sample_candidates(
    gp,
    acq_name: str,
    best_y: float,
    xi: float,
    kappa: float,
    z0: np.ndarray,
    n_steps: int,
    burn: int,
    thin: int,
    beta: float,
    tau: float,
    rng: np.random.Generator,
    # --- updates to match paper spirit ---
    adapt_beta: bool = True,
    beta_target: float = 0.243,
    beta_lr: float = 0.05,
    beta_min: float = 1e-3,
    beta_max: float = 0.999,
    # numerical safety for PI
    pi_eps: float = 1e-12,
):
    """
    Returns:
      samples: (M,d)      samples after burn-in, thinned
      chain:   (K,d)      thinned chain (for plotting)
      acq_chain: (K,)     "acq value" along chain (EI/UCB) or PI (not log)
      accept_rate: float  overall acceptance rate
      beta_trace: (K,)    beta values at stored chain points (useful to debug adaptation)
    """
    d = z0.shape[0]
    z = z0.copy()

    def gp_mu_std(z_in: np.ndarray):
        mu, std = gp.predict(z_in.reshape(1, -1), return_std=True)
        return float(mu[0]), float(std[0])

    # --- define "tilt" in log-space for MH ---
    def log_tilt(z_in: np.ndarray) -> float:
        mu, std = gp_mu_std(z_in)

        if acq_name == "ei":
            a = float(expected_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
            return tau * a

        if acq_name == "ucb":
            a = float(ucb(mu, std, kappa=kappa))
            return tau * a

        if acq_name == "pi":
            pi = float(probability_of_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
            # paper-like: likelihood tilt uses PI; use log for stability
            return tau * np.log(pi + pi_eps)

        raise ValueError("acq_name must be 'ei', 'ucb', or 'pi'")

    # --- value to record for plotting (not the MH energy) ---
    def record_val(z_in: np.ndarray) -> float:
        mu, std = gp_mu_std(z_in)
        if acq_name == "ei":
            return float(expected_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
        if acq_name == "ucb":
            return float(ucb(mu, std, kappa=kappa))
        if acq_name == "pi":
            return float(probability_of_improvement(np.array([mu]), np.array([std]), best_y=best_y, xi=xi)[0])
        raise ValueError("acq_name must be 'ei', 'ucb', or 'pi'")

    logw = log_tilt(z)
    accepted = 0

    samples = []
    chain = []
    acq_chain = []
    beta_trace = []

    for t in range(1, n_steps + 1):
        xi_n = rng.standard_normal(d)
        z_prop = np.sqrt(1.0 - beta**2) * z + beta * xi_n

        logw_prop = log_tilt(z_prop)
        log_alpha = logw_prop - logw

        # MH accept
        if np.log(rng.random()) < min(0.0, log_alpha):
            z = z_prop
            logw = logw_prop
            accepted += 1
            accepted_step = True
        else:
            accepted_step = False

        # Adaptive beta (paper spirit): update during burn-in only
        if adapt_beta and (t <= burn):
            # Use acceptance probability as in the paper pseudo-code
            alpha_prob = min(1.0, float(np.exp(min(0.0, log_alpha))))
            beta = float(np.clip(beta + beta_lr * (alpha_prob - beta_target), beta_min, beta_max))

        # store thinned chain for plotting
        if t % thin == 0:
            chain.append(z.copy())
            acq_chain.append(record_val(z))
            beta_trace.append(beta)

        # store samples after burn-in (also thinned)
        if t > burn and (t - burn) % thin == 0:
            samples.append(z.copy())

    samples = np.array(samples) if len(samples) else np.zeros((0, d))
    chain = np.array(chain) if len(chain) else np.zeros((0, d))
    acq_chain = np.array(acq_chain) if len(acq_chain) else np.zeros((0,))
    beta_trace = np.array(beta_trace) if len(beta_trace) else np.zeros((0,))

    accept_rate = accepted / max(1, n_steps)
    return samples, chain, acq_chain, accept_rate, beta_trace

# =========================
# Visualizations
# =========================
def plot_step(save_path,
              Z_data,           # (Nd,2) background
              Z_obs, y_obs,     # evaluated BO points
              z_next, y_next,
              z_best, y_best,
              chain,            # pCN chain (thin)
              cand,             # pCN samples
              grid_x, grid_y, A_grid, A_name,
              x_next, step_size,
              best_so_far,
              accept_rate):
    fig = plt.figure(figsize=(16, 5))

    # (1) Acquisition heatmap + overlays
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(grid_x, grid_y, A_grid, levels=35)
    plt.colorbar(im, ax=ax1, label=A_name)

    ax1.scatter(Z_data[:,0], Z_data[:,1], s=8, alpha=0.12, linewidth=0.0, label="data (encoded)")
    ax1.scatter(Z_obs[:,0], Z_obs[:,1], c=y_obs, s=55, edgecolor="black", linewidth=0.4, label="evaluated z")
    if chain.shape[0] > 0:
        ax1.plot(chain[:,0], chain[:,1], linewidth=1.2, alpha=0.9, label=f"pCN chain (acc={accept_rate:.2f})")
    if cand.shape[0] > 0:
        ax1.scatter(cand[:,0], cand[:,1], s=30, alpha=0.8, marker="x", label="pCN candidates")

    ax1.scatter([z_best[0]], [z_best[1]], c="red", s=260, marker="*", edgecolor="black", linewidth=1.0,
                label=f"best y={y_best:.2f}")
    ax1.scatter([z_next[0]], [z_next[1]], c="yellow", s=160, marker="D", edgecolor="black", linewidth=1.0,
                label=f"next y={y_next:.2f}")

    ax1.set_title("COWBOYS: acquisition + pCN candidates")
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
                label="best BO z")
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
# Main (COWBOYS-faithful)
#   - Initial design: sample z ~ N(0,I), decode x = h(z), evaluate y=f(x)
#   - Surrogate GP is fit in STRUCTURE/SEQUENCE space x (not in latent z)
#   - pCN runs in latent z, but the MH "likelihood tilt" is PI(x=h(z)) under the GP
#   - Optional: choose next point from the sampled set using EI/UCB (qEI for batch=1 ≈ EI)
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=60)

    # Utility used to choose x_next from the pCN sample set (paper uses qEI for batch; for batch=1 EI is standard)
    ap.add_argument("--acq", type=str, default="ei", choices=["ei", "ucb", "pi"])
    ap.add_argument("--kappa", type=float, default=2.2)   # UCB
    ap.add_argument("--xi", type=float, default=0.00)     # EI/PI margin (use 0.0 to match threshold f*)

    # pCN parameters (sampling in latent)
    ap.add_argument("--pcn_steps", type=int, default=2500)
    ap.add_argument("--pcn_burn", type=int, default=800)
    ap.add_argument("--pcn_thin", type=int, default=10)
    ap.add_argument("--pcn_beta", type=float, default=0.35)
    ap.add_argument("--tau", type=float, default=40.0)    # PI tilt temperature (bigger => more concentrated)
    ap.add_argument("--adapt_beta", action="store_true")   # closer to paper sampler
    ap.add_argument("--beta_target", type=float, default=0.243)

    # plotting grid in latent (we still visualize on z, but acquisition is computed via decode->GP-on-x)
    ap.add_argument("--z_box", type=float, default=5.0)
    ap.add_argument("--grid_res", type=int, default=140)

    # background data scatter (for plot only)
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    # fallback if pCN returns zero samples (rare but possible)
    ap.add_argument("--fallback_random_tries", type=int, default=50)

    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generation first."

    data = np.load(data_npz, allow_pickle=True)
    X_all = data["X"].astype(np.float64)       # only used for plotting cloud (optional)
    target = data["target"].astype(np.float64)
    # (types, y_data) are no longer used to initialize, to match COWBOYS

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
    assert os.path.exists(vae_path), f"Missing VAE at {vae_path}. Train it first."
    vae, L, latent_dim, train_types = load_vae(vae_path, device)
    print(f"Loaded VAE: L={L}, latent_dim={latent_dim}, train_types={train_types}")
    assert latent_dim == 2, "This script assumes latent_dim=2 for 2D visualizations."
    assert X_all.shape[1] == L, f"Dataset L={X_all.shape[1]} but VAE expects L={L}"

    # output dirs
    plot_root = os.path.join(outdir, "comparison")
    plot_root = os.path.join(plot_root, "cowboys")
    step_dir = os.path.join(plot_root, "steps")
    ensure_dir(step_dir)

    # --- batched decode helper (fast for grids/candidates)
    def decode_batch_z_to_x(Z: np.ndarray) -> np.ndarray:
        """
        Z: (N,2) numpy
        returns X_radians: (N,L) numpy
        """
        Zt = torch.tensor(Z, dtype=torch.float32, device=device)
        with torch.no_grad():
            x_hat_scaled = vae.decode(Zt)  # (N,L) in [-1,1]
        return (x_hat_scaled.detach().cpu().numpy().astype(np.float64) * np.pi)

    # background latent cloud (for plotting only)
    Z_data = encode_dataset_to_z(
        vae, X_all, device,
        n_points=args.n_data_scatter,
        seed=args.data_scatter_seed
    )

    # ============================================================
    # COWBOYS initial design (Algorithm 2, eq (2)):
    #   z ~ N(0,I), x = h(z), y = f(x)
    # Dataset for GP is in STRUCTURE space: D_n^X = {(x_i, y_i)}
    # ============================================================
    Z_obs, X_obs, y_obs = [], [], []
    for _ in range(args.n_init):
        z0 = rng.standard_normal(latent_dim)
        x0 = decode_z_to_delta_theta(vae, z0, device)       # x0 in radians (sequence space)
        y0 = float(oracle_f(x0, target, step_size, w_close, w_smooth))
        Z_obs.append(z0); X_obs.append(x0); y_obs.append(y0)

    Z_obs = np.asarray(Z_obs, dtype=float)          # (n,2)
    X_obs = np.asarray(X_obs, dtype=float)          # (n,L)
    y_obs = np.asarray(y_obs, dtype=float)          # (n,)

    print("[INIT] best y:", float(y_obs.max()))
    best_so_far = [float(y_obs.max())]

    # ============================================================
    # BO loop (Algorithm 2, sequential):
    #   1) Fit structured GP on (x,y)
    #   2) Let f* be best observed
    #   3) Sample z from p(z | f(h(z)) > f*, D) using pCN
    #      (practically: target density ∝ PI(h(z))^tau * N(0,I))
    #   4) Decode x=h(z), evaluate y=f(x), append to D
    # ============================================================
    for t in range(1, args.n_steps + 1):

        # ---- Fit GP surrogate in SEQUENCE/STRUCTURE space x (matches COWBOYS)
        # Kernel in x-space (continuous):
        #   Constant * Matern(ARD) + White
        kernel_x = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(L), nu=2.5, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )

        gp_x = GaussianProcessRegressor(
            kernel=kernel_x,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=args.seed
        )
        gp_x.fit(X_obs, y_obs)

        best_y = float(y_obs.max())
        best_idx = int(np.argmax(y_obs))
        z_best = Z_obs[best_idx].copy()
        y_best = float(y_obs[best_idx])

        # ---- Wrap GP so pCN can call gp.predict(z) but it actually predicts on decoded x=h(z)
        class GPOnDecoded:
            def __init__(self, gp_struct, decode_batch_fn):
                self.gp_struct = gp_struct
                self.decode_batch_fn = decode_batch_fn
            def predict(self, Z_in, return_std=True):
                Z_in = np.asarray(Z_in, dtype=float)
                X_in = self.decode_batch_fn(Z_in)  # (N,L)
                return self.gp_struct.predict(X_in, return_std=return_std)

        gp_decoded = GPOnDecoded(gp_x, decode_batch_z_to_x)

        # ---- Visualization grid in latent: A(z) computed via decode->GP(x)
        z1 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        z2 = np.linspace(-args.z_box, args.z_box, args.grid_res)
        grid_x, grid_y = np.meshgrid(z1, z2)
        Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        mu_g, std_g = gp_decoded.predict(Z_grid, return_std=True)
        # For plotting, show the chosen "utility" (EI/UCB/PI) evaluated at x=h(z)
        if args.acq == "ei":
            A = expected_improvement(mu_g, std_g, best_y=best_y, xi=args.xi)
            A_name = "EI(x=h(z))"
        elif args.acq == "ucb":
            A = ucb(mu_g, std_g, kappa=args.kappa)
            A_name = "UCB(x=h(z))"
        else:
            A = probability_of_improvement(mu_g, std_g, best_y=best_y, xi=args.xi)
            A_name = "PI(x=h(z))"
        A_grid = A.reshape(args.grid_res, args.grid_res)

        # ============================================================
        # COWBOYS sampling step (Algorithms 3–4 idea, sequential case):
        # Target: pi(z) ∝ PI(h(z))^tau * N(0,I)
        # This is achieved by calling pcn_sample_candidates with acq_name="pi"
        # on the "decoded GP" wrapper.
        # ============================================================
        pcn_res = pcn_sample_candidates(
            gp=gp_decoded,
            acq_name="pi",          # IMPORTANT: match COWBOYS conditioning event via PI
            best_y=best_y,
            xi=args.xi,             # threshold margin (0.0 -> strict f*)
            kappa=args.kappa,       # unused for PI, but keep signature
            z0=z_best,
            n_steps=args.pcn_steps,
            burn=args.pcn_burn,
            thin=args.pcn_thin,
            beta=args.pcn_beta,
            tau=args.tau,
            rng=rng,
            adapt_beta=args.adapt_beta,
            beta_target=args.beta_target,
        )
        cand_mcmc, chain, acq_chain, acc_rate = pcn_res[:4]

        # If pCN yields no samples (can happen with extreme settings), fallback to random prior draws.
        if cand_mcmc.shape[0] == 0:
            cand_mcmc = rng.standard_normal(size=(max(10, args.fallback_random_tries), latent_dim))

        # ---- Choose the next point from the pCN sample set using a "utility" (paper uses qEI for batches)
        # For batch=1, EI is the classic choice.
        X_cand = decode_batch_z_to_x(cand_mcmc)
        mu_c, std_c = gp_x.predict(X_cand, return_std=True)

        if args.acq == "ei":
            util = expected_improvement(mu_c, std_c, best_y=best_y, xi=args.xi)
        elif args.acq == "ucb":
            util = ucb(mu_c, std_c, kappa=args.kappa)
        else:
            util = probability_of_improvement(mu_c, std_c, best_y=best_y, xi=args.xi)

        pick_idx = int(np.argmax(util))
        z_next = cand_mcmc[pick_idx]
        x_next = X_cand[pick_idx]

        # ---- Evaluate true oracle on x_next (expensive black box)
        y_next = float(oracle_f(x_next, target, step_size, w_close, w_smooth))

        # ---- Update datasets (GP stays in X-space, but we also store Z for z_best initialization + plotting)
        Z_obs = np.vstack([Z_obs, z_next.reshape(1, -1)])
        X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
        y_obs = np.append(y_obs, y_next)

        best_so_far.append(float(y_obs.max()))

        if t == 1 or t % 5 == 0:
            print(f"[COWBOYS] step {t:3d} | y_next={y_next: .4f} | best={float(y_obs.max()): .4f} | acc={acc_rate:.2f}")

        # ---- Per-step plot (now: acquisition is computed via decode->GP(x))
        save_path = os.path.join(step_dir, f"step_{t:03d}.png")
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

    # final summary
    best_idx = int(np.argmax(y_obs))
    z_best = Z_obs[best_idx]
    x_best = X_obs[best_idx]
    y_best = float(y_obs[best_idx])

    final_png = os.path.join(plot_root, "final_summary.png")
    plot_final_summary(final_png, Z_data, Z_obs, y_obs, z_best, x_best, y_best, step_size)

    # save trace (now includes X_obs too, since GP is fit in x-space)
    trace_path = os.path.join(plot_root, "trace.npz")
    np.savez_compressed(trace_path, Z_obs=Z_obs, X_obs=X_obs, y_obs=y_obs, best_so_far=np.array(best_so_far))

    print("\n=== COWBOYS finished (structure-space GP + PI-tilted pCN) ===")
    print("Best oracle:", y_best)
    print("Best z:", z_best)
    print("Saved frames:", os.path.abspath(step_dir))
    print("Saved final summary:", os.path.abspath(final_png))
    print("Saved trace:", os.path.abspath(trace_path))


if __name__ == "__main__":
    main()
