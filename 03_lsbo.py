# Benchmark default:
# python 03_lsbo.py --outdir toy_circle_data --seed 0 --n_init 15 --n_steps 150 --n_cand 200 --xi 0.08 --z_box 5 --selection_rule argmax_ei --proposal_dist trunc_normal --eps_random 0.0 --topk_ei 1


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
    oracle_f,
    set_seed,
    turtle_path,
)
from toy_diagnostics import ToyDiagnosticsLogger, latent_box_diversity_normalizer


# =========================
# Plotting
# =========================
def plot_step_2d(step_idx, save_path,Z_data,
                 grid_x, grid_y, EI_grid,
                 Z_obs, y_obs, z_next,
                 Z_cand,
                 x_next, y_next,
                 best_so_far,
                 step_size):
    fig = plt.figure(figsize=(14, 4.8))

    # Visualization-only EI heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(grid_x, grid_y, EI_grid, levels=30)
    plt.colorbar(im, ax=ax1, label="EI")
    ax1.scatter(Z_data[:,0], Z_data[:,1], s=8, alpha=0.12, linewidth=0.0, label="data (encoded)")
    if Z_cand is not None and Z_cand.shape[0] > 0:
        ax1.scatter(Z_cand[:,0], Z_cand[:,1], s=18, alpha=0.45, marker="x", label="EI candidates")
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


def sample_latent_candidates(n_cand, latent_dim, z_box, proposal_dist, rng):
    if n_cand <= 0:
        raise ValueError("--n_cand must be positive.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if z_box <= 0:
        raise ValueError("--z_box must be positive.")

    if proposal_dist == "normal":
        return rng.standard_normal(size=(n_cand, latent_dim))

    if proposal_dist == "uniform_box":
        return rng.uniform(-z_box, z_box, size=(n_cand, latent_dim))

    if proposal_dist == "normal_uniform_mixture":
        n_uniform = n_cand // 3
        n_normal = n_cand - n_uniform
        parts = [rng.standard_normal(size=(n_normal, latent_dim))]
        if n_uniform > 0:
            parts.append(rng.uniform(-z_box, z_box, size=(n_uniform, latent_dim)))
        return np.vstack(parts)

    if proposal_dist == "trunc_normal":
        parts = []
        n_kept = 0
        while n_kept < n_cand:
            batch_n = max(1024, 2 * (n_cand - n_kept))
            batch = rng.standard_normal(size=(batch_n, latent_dim))
            keep = batch[np.all(np.abs(batch) <= z_box, axis=1)]
            if keep.shape[0] > 0:
                parts.append(keep)
                n_kept += keep.shape[0]
        return np.vstack(parts)[:n_cand]

    if proposal_dist == "sobol_box":
        try:
            from scipy.stats import qmc

            seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            sampler = qmc.Sobol(d=latent_dim, scramble=True, seed=seed)
            m = int(np.ceil(np.log2(n_cand)))
            unit = sampler.random_base2(m=m)[:n_cand]
            return -z_box + 2.0 * z_box * unit
        except Exception:
            print("[WARN] scipy Sobol unavailable; falling back to uniform_box candidates.")
            return rng.uniform(-z_box, z_box, size=(n_cand, latent_dim))

    raise ValueError(f"Unknown proposal_dist: {proposal_dist}")


def is_non_duplicate(z, Z_obs, duplicate_z_tol):
    if Z_obs.shape[0] == 0:
        return True
    return bool(np.min(np.linalg.norm(Z_obs - z.reshape(1, -1), axis=1)) > duplicate_z_tol)


def select_candidate_index(Zcand, ei, Z_obs, duplicate_z_tol, selection_rule, topk_ei, eps_random, rng):
    score = np.where(np.isfinite(ei), ei, -np.inf)
    order = np.argsort(score)[::-1]
    best_idx = int(order[0])

    non_duplicate_order = [
        int(idx)
        for idx in order
        if is_non_duplicate(Zcand[int(idx)], Z_obs, duplicate_z_tol)
    ]
    if not non_duplicate_order:
        return best_idx, True

    if selection_rule == "argmax_ei":
        return int(non_duplicate_order[0]), False

    if selection_rule == "topk_random":
        k = min(max(1, int(topk_ei)), len(non_duplicate_order))
        return int(rng.choice(non_duplicate_order[:k])), False

    if selection_rule == "epsilon_topk":
        if rng.random() < eps_random:
            return int(rng.choice(non_duplicate_order)), False
        k = min(max(1, int(topk_ei)), len(non_duplicate_order))
        return int(rng.choice(non_duplicate_order[:k])), False

    raise ValueError(f"Unknown selection_rule: {selection_rule}")


# =========================
# Main LSBO
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument(
        "--plotroot",
        type=str,
        default=None,
        help="Where to save plots/traces. Supports {seed} and {method}; default is results/toy_runs/lsbo/seed_{seed}.",
    )
    ap.add_argument("--vae_path", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=150)

    ap.add_argument("--xi", type=float, default=0.08)
    ap.add_argument("--eps_random", type=float, default=0.0)   # used only by --selection_rule epsilon_topk
    ap.add_argument("--topk_ei", type=int, default=1)
    ap.add_argument("--n_cand", type=int, default=200,
                    help="Latent proposals scored by EI per BO iteration.")
    ap.add_argument("--selection_rule", type=str, default="argmax_ei",
                    choices=["argmax_ei", "topk_random", "epsilon_topk"])
    ap.add_argument("--proposal_dist", type=str, default="trunc_normal",
                    choices=["trunc_normal", "normal", "uniform_box", "sobol_box", "normal_uniform_mixture"])
    ap.add_argument("--duplicate_z_tol", type=float, default=1e-6)
    ap.add_argument("--init_design_path", type=str, default=None,
                    help="Optional shared initial design .npz containing Z_init, X_init, and y_init.")

    ap.add_argument("--z_box", type=float, default=5.0)
    ap.add_argument("--grid_res", type=int, default=140)
    
    ap.add_argument("--n_data_scatter", type=int, default=3000,
                help="How many dataset points to encode+plot in latent space for final_summary.")
    ap.add_argument("--data_scatter_seed", type=int, default=0)

    ap.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True,
                    help="Export non-invasive toy diagnostics JSON for post-processing.")
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics")
    ap.add_argument("--diagnostics_top_k", type=int, default=10)
    ap.add_argument("--diagnostics_max_proposals", type=int, default=2000)
    ap.add_argument("--diagnostics_background_res", type=int, default=60,
                    help="2D latent grid resolution for diagnostic-only oracle background; 0 disables it.")
    args = ap.parse_args()
    if args.n_cand <= 0:
        raise ValueError("--n_cand must be positive.")
    if args.topk_ei <= 0:
        raise ValueError("--topk_ei must be positive.")
    if not (0.0 <= args.eps_random <= 1.0):
        raise ValueError("--eps_random must lie in [0, 1].")
    if args.duplicate_z_tol < 0.0:
        raise ValueError("--duplicate_z_tol must be non-negative.")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing {data_npz}. Run data generation first."

    # Load dataset
    data = np.load(data_npz, allow_pickle=True)
    X = data["X"].astype(np.float64)        # (N,L) radians
    target = data["target"].astype(np.float64)

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

    diagnostics = None
    if args.diagnostics:
        diagnostics = ToyDiagnosticsLogger(
            method="lsbo",
            seed=args.seed,
            problem="toy_circle_path",
            config=vars(args),
            results_root=args.diagnostics_root,
            top_k=args.diagnostics_top_k,
            max_proposals=args.diagnostics_max_proposals,
            diversity_normalizer=latent_box_diversity_normalizer(args.z_box, latent_dim),
            step_size=step_size,
            target_path=target,
            decode_latent_fn=lambda z: decode_z_to_delta_theta(vae, z, device),
            objective_fn=lambda x: oracle_f(x, target, step_size, w_close, w_smooth),
        )
        if latent_dim == 2 and args.diagnostics_background_res > 1:
            diagnostics.set_objective_background(args.z_box, args.diagnostics_background_res)

    # Output dirs. Seed-aware by default so repeated runs do not overwrite each other.
    plotroot_template = args.plotroot or os.path.join("results", "toy_runs", "lsbo", "seed_{seed}")
    plot_root = plotroot_template.format(seed=args.seed, method="lsbo")
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

    # Initial observations are expensive oracle calls. For final cross-method
    # comparisons, provide the same --init_design_path to every method.
    if args.init_design_path is not None:
        assert os.path.exists(args.init_design_path), f"Missing init design at {args.init_design_path}"
        init_data = np.load(args.init_design_path, allow_pickle=False)
        for key in ("Z_init", "X_init", "y_init"):
            if key not in init_data:
                raise KeyError(f"{args.init_design_path} must contain {key}")
        Z_obs = init_data["Z_init"].astype(float)
        X_init = init_data["X_init"].astype(float)
        y_obs = init_data["y_init"].astype(float).reshape(-1)
        if Z_obs.ndim != 2 or Z_obs.shape[1] != latent_dim:
            raise ValueError(f"Z_init must have shape (n, {latent_dim})")
        if X_init.ndim != 2 or X_init.shape[1] != L:
            raise ValueError(f"X_init must have shape (n, {L})")
        if Z_obs.shape[0] != X_init.shape[0] or y_obs.shape[0] != Z_obs.shape[0]:
            raise ValueError("Z_init, X_init, and y_init must have the same length")
        x_obs = [x.copy() for x in X_init]
        print(f"[INIT] loaded shared initial design: {args.init_design_path}")
    else:
        print("[WARN] Using fallback random LSBO initialization. Use --init_design_path for final cross-method comparisons.")
        Z_init, x_obs, y_init = [], [], []
        for _ in range(args.n_init):
            z0 = rng.standard_normal(latent_dim)
            x0 = decode_z_to_delta_theta(vae, z0, device)
            y0 = oracle_f(x0, target, step_size, w_close, w_smooth)
            Z_init.append(z0); x_obs.append(x0); y_init.append(y0)
        Z_obs = np.asarray(Z_init, dtype=float)
        y_obs = np.asarray(y_init, dtype=float)

    n_init_actual = int(Z_obs.shape[0])
    if n_init_actual <= 0:
        raise ValueError("Initial design must contain at least one observation.")
    if diagnostics is not None:
        diagnostics.set_initial_observations(Z_obs, y_obs, np.asarray(x_obs, dtype=float))

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
        proposal_latents = None
        proposal_acquisition_values = None

        # ---- Propose z_next.
        # Internal proposal/EI evaluations are optimizer work, not oracle calls.
        # This benchmark is fair in oracle budget, not necessarily wall-clock time.
        grid_x = grid_y = EI_grid = None
        if latent_dim == 2:
            # Grid EI is visualization-only; the selected point comes from M proposals below.
            z1 = np.linspace(-args.z_box, args.z_box, args.grid_res)
            z2 = np.linspace(-args.z_box, args.z_box, args.grid_res)
            grid_x, grid_y = np.meshgrid(z1, z2)
            Z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

            mu_grid, std_grid = gp.predict(Z_grid, return_std=True)
            EI_grid = expected_improvement(mu_grid, std_grid, best_y=best_y, xi=args.xi).reshape(args.grid_res, args.grid_res)

        Zcand = sample_latent_candidates(
            n_cand=args.n_cand,
            latent_dim=latent_dim,
            z_box=args.z_box,
            proposal_dist=args.proposal_dist,
            rng=rng,
        )
        mu_cand, std_cand = gp.predict(Zcand, return_std=True)
        ei_cand = expected_improvement(mu_cand, std_cand, best_y=best_y, xi=args.xi)
        proposal_latents = Zcand
        proposal_acquisition_values = ei_cand

        pick_idx, duplicate_fallback = select_candidate_index(
            Zcand=Zcand,
            ei=ei_cand,
            Z_obs=Z_obs,
            duplicate_z_tol=args.duplicate_z_tol,
            selection_rule=args.selection_rule,
            topk_ei=args.topk_ei,
            eps_random=args.eps_random,
            rng=rng,
        )
        z_next = Zcand[pick_idx]
        selected_ei = float(ei_cand[pick_idx])
        max_ei = float(np.max(ei_cand))
        if duplicate_fallback:
            print(f"[WARN] step {t}: all candidates were within duplicate_z_tol; using max-EI duplicate fallback.")

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
        if diagnostics is not None:
            best_idx_diag = int(np.argmax(y_obs))
            diagnostics.record_iteration(
                iteration=t,
                selected_latent=z_next,
                selected_objective=float(y_next),
                best_so_far_objective=float(y_obs[best_idx_diag]),
                incumbent_best_latent=Z_obs[best_idx_diag],
                incumbent_best_objective=float(y_obs[best_idx_diag]),
                selected_sequence=x_next,
                incumbent_best_sequence=np.asarray(x_obs[best_idx_diag], dtype=float),
                proposal_latents=proposal_latents,
                proposal_acquisition_values=proposal_acquisition_values,
                extra={
                    "selection_acquisition": "expected_improvement",
                    "selection_rule": args.selection_rule,
                    "proposal_dist": args.proposal_dist,
                    "duplicate_fallback": bool(duplicate_fallback),
                    "selected_ei": selected_ei,
                    "max_ei": max_ei,
                    "n_cand": int(proposal_latents.shape[0]) if proposal_latents is not None else 0,
                },
            )

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
                Z_cand=proposal_latents,
                x_next=x_next, y_next=y_next,
                best_so_far=best_so_far,
                step_size=step_size
            )

    # ----- Save trace
    trace_path = os.path.join(plot_root, "bo_trace.npz")
    oracle_calls = np.arange(n_init_actual, n_init_actual + args.n_steps + 1)
    np.savez_compressed(
        trace_path,
        Z_obs=Z_obs,
        y_obs=y_obs,
        best_so_far=np.array(best_so_far, dtype=float),
        oracle_calls=oracle_calls,
    )

    best_idx = int(np.argmax(y_obs))
    z_best = Z_obs[best_idx]
    x_best = np.asarray(x_obs[best_idx], dtype=float)
    y_best = float(y_obs[best_idx])

    # Final summary plot (only if 2D latent)
    final_png = None
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

    if final_png is not None:
        print("Saved final summary:", os.path.abspath(final_png))
    if diagnostics is not None:
        diagnostics_path = diagnostics.finalize(
            extra_summary={
                "best_objective": y_best,
                "best_latent": z_best,
                "trace_path": trace_path,
                "plot_root": plot_root,
            }
        )
        print("Saved toy diagnostics:", os.path.abspath(diagnostics_path))
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
