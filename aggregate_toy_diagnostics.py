"""Aggregate and plot toy Bayesian-optimization diagnostics.

Run after one or more experiment scripts have produced
results/toy_diagnostics/raw/<method>/<problem>/seed_<n>/metrics.json.
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "lsbo": "LSBO",
    "cowboys": "COWBOYS",
    "dgbo_latent_diffusion": "DGBO",
    "dgbo_latent_diffusion_distillation": "DGBO Distill",
    "cowboys_flow_latent": "COWBOYS Flow",
}

METHOD_COLORS = {
    "lsbo": "tab:blue",
    "cowboys": "tab:orange",
    "dgbo_latent_diffusion": "tab:purple",
    "dgbo_latent_diffusion_distillation": "tab:green",
    "cowboys_flow_latent": "tab:red",
}

MAIN_METRICS = [
    ("best_so_far_objective", "Best-so-far objective"),
    ("nearest_previous_selected_latent_distance", "Nearest previous selected latent distance"),
    ("adjusted_top_k_objective", "Adjusted top-k objective"),
]

RAW_TOPK_METRICS = [
    ("mean_top_k_objective", "Mean top-k objective"),
    ("top_k_latent_diversity", "Top-k latent diversity"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def finite_float(value):
    if value is None:
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def arr2d(value):
    if value is None:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 2), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def center_path(path):
    pts = arr2d(path)
    if pts.shape[0] == 0:
        return pts
    return pts - pts.mean(axis=0, keepdims=True)


def objective_background(run):
    bg = run.get("objective_background") or {}
    try:
        z1 = np.asarray(bg.get("z1"), dtype=np.float64)
        z2 = np.asarray(bg.get("z2"), dtype=np.float64)
        grid = np.asarray(bg.get("objective_grid"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if z1.ndim != 1 or z2.ndim != 1 or grid.shape != (z2.size, z1.size):
        return None
    if z1.size < 2 or z2.size < 2:
        return None
    return z1, z2, grid, str(bg.get("label", "oracle f(decode(z))"))


def draw_objective_background(ax, run, filled=False):
    bg = objective_background(run)
    if bg is None:
        return None
    z1, z2, grid, label = bg
    if filled:
        im = ax.contourf(z1, z2, grid, levels=28, cmap="viridis", alpha=0.36)
        return im, label
    ax.contour(z1, z2, grid, levels=10, colors="0.35", linewidths=0.55, alpha=0.42)
    return None


def load_runs(results_root):
    pattern = os.path.join(results_root, "raw", "*", "*", "seed_*", "metrics.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = path
        payload["_run_dir"] = os.path.dirname(path)
        runs.append(payload)
    return runs


def series(run, key):
    metric_series = run.get("metric_series", {})
    iterations = metric_series.get("bo_iteration", [])
    values = metric_series.get(key, [])
    n = min(len(iterations), len(values))
    x = np.asarray(iterations[:n], dtype=np.int64)
    y = np.asarray([finite_float(v) for v in values[:n]], dtype=np.float64)
    return x, y


def aggregate_series(runs, key):
    all_iters = sorted({int(i) for run in runs for i in series(run, key)[0]})
    if not all_iters:
        return np.zeros((0,), dtype=int), np.zeros((0,)), np.zeros((0,))

    mat = np.full((len(runs), len(all_iters)), np.nan, dtype=np.float64)
    pos = {it: j for j, it in enumerate(all_iters)}
    for r, run in enumerate(runs):
        x, y = series(run, key)
        for it, val in zip(x, y):
            mat[r, pos[int(it)]] = val

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return np.asarray(all_iters, dtype=int), mean, std


def plot_metric_grid(method_runs, metric_specs, save_path, title):
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.4 * len(metric_specs), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, metric_specs):
        for method, runs in method_runs.items():
            x, mean, std = aggregate_series(runs, key)
            if x.size == 0 or np.all(~np.isfinite(mean)):
                continue
            color = METHOD_COLORS.get(method, None)
            ax.plot(x, mean, linewidth=2.0, label=METHOD_LABELS.get(method, method), color=color)
            if len(runs) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_title(label)
        ax.set_xlabel("BO iteration")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("metric value")
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def representative_rows(run, requested):
    rows = run.get("iterations", [])
    if not rows:
        return []
    chosen = []
    seen = set()
    available = np.asarray([int(row.get("iteration", 0)) for row in rows], dtype=int)
    for req in requested:
        idx = int(np.argmin(np.abs(available - int(req))))
        it = int(rows[idx].get("iteration", 0))
        if it not in seen:
            chosen.append(rows[idx])
            seen.add(it)
    return chosen


def first_run_by_method(method_runs):
    out = {}
    for method, runs in method_runs.items():
        out[method] = sorted(runs, key=lambda r: int(r.get("seed", 0)))[0]
    return out


def latent_limits_for_run(run):
    pieces = []
    pieces.append(arr2d(run.get("initial_observations", {}).get("latents")))
    for row in run.get("iterations", []):
        pieces.append(arr2d(row.get("selected_latent")))
        pieces.append(arr2d(row.get("incumbent_best_latent")))
        prop = arr2d(row.get("proposal_latents"))
        if prop.shape[0] > 0:
            take = min(200, prop.shape[0])
            idx = np.linspace(0, prop.shape[0] - 1, num=take, dtype=int)
            pieces.append(prop[idx])
    pts = np.vstack([p for p in pieces if p.shape[0] > 0]) if pieces else np.zeros((0, 2))
    if pts.shape[0] == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    lo = np.nanquantile(pts, 0.02, axis=0)
    hi = np.nanquantile(pts, 0.98, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    return (float(lo[0] - 0.15 * span[0]), float(hi[0] + 0.15 * span[0])), (
        float(lo[1] - 0.15 * span[1]),
        float(hi[1] + 0.15 * span[1]),
    )


def plot_latent_trajectories(method_runs, save_path):
    samples = first_run_by_method(method_runs)
    n = len(samples)
    if n == 0:
        return
    cols = min(2, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.5 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for ax, (method, run) in zip(axes.ravel(), samples.items()):
        bg_artist = draw_objective_background(ax, run, filled=True)
        init_z = arr2d(run.get("initial_observations", {}).get("latents"))
        selected = arr2d([row.get("selected_latent") for row in run.get("iterations", [])])
        if init_z.shape[0] > 0:
            ax.scatter(init_z[:, 0], init_z[:, 1], s=24, alpha=0.28, color="0.55", label="initial")
        if selected.shape[0] > 0:
            t = np.arange(1, selected.shape[0] + 1)
            ax.plot(selected[:, 0], selected[:, 1], color="0.25", linewidth=1.2, alpha=0.75)
            sc = ax.scatter(selected[:, 0], selected[:, 1], c=t, cmap="viridis", s=42, edgecolor="black", linewidth=0.25)
            ax.scatter(selected[0, 0], selected[0, 1], s=90, marker="o", facecolor="white", edgecolor="black", label="first BO")
            ax.scatter(selected[-1, 0], selected[-1, 1], s=130, marker="*", color="red", edgecolor="black", label="last BO")
            plt.colorbar(sc, ax=ax, label="iteration")
        elif bg_artist is not None:
            plt.colorbar(bg_artist[0], ax=ax, label=bg_artist[1])
        zlim = latent_limits_for_run(run)
        ax.set_xlim(zlim[0])
        ax.set_ylim(zlim[1])
        ax.set_aspect("equal", "box")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_proposal_clouds(run, requested, save_path):
    rows = representative_rows(run, requested)
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(5.4 * len(rows), 4.8), squeeze=False)
    zlim = latent_limits_for_run(run)
    for ax, row in zip(axes.ravel(), rows):
        draw_objective_background(ax, run, filled=False)
        prop = arr2d(row.get("proposal_latents"))
        acq = row.get("proposal_acquisition_values") or []
        acq = np.asarray([finite_float(v) for v in acq], dtype=np.float64)
        if prop.shape[0] > 0:
            if acq.shape[0] == prop.shape[0] and np.any(np.isfinite(acq)):
                sc = ax.scatter(prop[:, 0], prop[:, 1], c=acq, cmap="magma", s=18, alpha=0.55, linewidth=0.0)
                plt.colorbar(sc, ax=ax, label="acq")
            else:
                ax.scatter(prop[:, 0], prop[:, 1], s=18, alpha=0.5, linewidth=0.0, color="tab:blue")

        sel = arr2d(row.get("selected_latent"))
        best = arr2d(row.get("incumbent_best_latent"))
        if best.shape[0] > 0:
            ax.scatter(best[:, 0], best[:, 1], marker="*", s=190, color="red", edgecolor="black", label="incumbent")
        if sel.shape[0] > 0:
            ax.scatter(sel[:, 0], sel[:, 1], marker="D", s=95, color="yellow", edgecolor="black", label="selected")
        ax.set_title(f"iteration {row.get('iteration')}")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_xlim(zlim[0])
        ax.set_ylim(zlim[1])
        ax.set_aspect("equal", "box")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.18)
    fig.suptitle(f"{METHOD_LABELS.get(run.get('method'), run.get('method'))}: proposal clouds")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_decoded_paths(run, requested, save_path):
    rows = representative_rows(run, requested)
    if not rows:
        return
    final_row = run.get("iterations", [])[-1]
    panels = rows + [final_row]
    labels = [f"iteration {row.get('iteration')}" for row in rows] + ["final incumbent"]

    target = center_path(run.get("target_path"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.4), squeeze=False)
    all_paths = [target]
    for row in rows:
        all_paths.append(center_path(row.get("selected_decoded_path")))
    all_paths.append(center_path(final_row.get("incumbent_best_decoded_path")))
    stacked = np.vstack([p for p in all_paths if p.shape[0] > 0]) if all_paths else np.zeros((0, 2))
    if stacked.shape[0] > 0:
        lo = np.min(stacked, axis=0)
        hi = np.max(stacked, axis=0)
        span = np.maximum(hi - lo, 1e-3)
        xlim = (float(lo[0] - 0.12 * span[0]), float(hi[0] + 0.12 * span[0]))
        ylim = (float(lo[1] - 0.12 * span[1]), float(hi[1] + 0.12 * span[1]))
    else:
        xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)

    for ax, row, label in zip(axes.ravel(), panels, labels):
        pts = center_path(row.get("incumbent_best_decoded_path") if label == "final incumbent" else row.get("selected_decoded_path"))
        if target.shape[0] > 0:
            ax.plot(target[:, 0], target[:, 1], color="black", linestyle="--", linewidth=1.5, alpha=0.75, label="target")
        if pts.shape[0] > 0:
            ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=2.3, label="decoded")
            ax.scatter([pts[0, 0]], [pts[0, 1]], s=36, color="tab:blue")
        ax.set_title(label)
        ax.set_aspect("equal", "box")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        ax.legend(loc="best")

    fig.suptitle(f"{METHOD_LABELS.get(run.get('method'), run.get('method'))}: decoded paths")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_run_gif(run, save_path, max_frames=80, fps=4):
    rows = run.get("iterations", [])
    if not rows:
        return False
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return False

    idx = np.arange(len(rows))
    if len(rows) > max_frames:
        idx = np.unique(np.linspace(0, len(rows) - 1, num=max_frames, dtype=int))
    frame_rows = [rows[i] for i in idx]
    zlim = latent_limits_for_run(run)
    target = center_path(run.get("target_path"))

    path_pieces = [target]
    for row in frame_rows:
        path_pieces.append(center_path(row.get("selected_decoded_path")))
    stacked = np.vstack([p for p in path_pieces if p.shape[0] > 0]) if path_pieces else np.zeros((0, 2))
    if stacked.shape[0] > 0:
        lo = np.min(stacked, axis=0)
        hi = np.max(stacked, axis=0)
        span = np.maximum(hi - lo, 1e-3)
        xlim = (float(lo[0] - 0.12 * span[0]), float(hi[0] + 0.12 * span[0]))
        ylim = (float(lo[1] - 0.12 * span[1]), float(hi[1] + 0.12 * span[1]))
    else:
        xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)

    fig, (ax_z, ax_path) = plt.subplots(1, 2, figsize=(10.2, 4.8))

    def draw(frame_i):
        row = frame_rows[frame_i]
        ax_z.clear()
        ax_path.clear()
        draw_objective_background(ax_z, run, filled=False)

        prop = arr2d(row.get("proposal_latents"))
        if prop.shape[0] > 0:
            ax_z.scatter(prop[:, 0], prop[:, 1], s=16, alpha=0.35, color="tab:blue", linewidth=0.0, label="proposal")

        hist = arr2d([r.get("selected_latent") for r in rows[: idx[frame_i] + 1]])
        if hist.shape[0] > 0:
            ax_z.plot(hist[:, 0], hist[:, 1], color="0.25", linewidth=1.3, alpha=0.85)
            ax_z.scatter(hist[:, 0], hist[:, 1], c=np.arange(hist.shape[0]), cmap="viridis", s=28)

        sel = arr2d(row.get("selected_latent"))
        best = arr2d(row.get("incumbent_best_latent"))
        if best.shape[0] > 0:
            ax_z.scatter(best[:, 0], best[:, 1], marker="*", s=180, color="red", edgecolor="black", label="incumbent")
        if sel.shape[0] > 0:
            ax_z.scatter(sel[:, 0], sel[:, 1], marker="D", s=90, color="yellow", edgecolor="black", label="selected")
        ax_z.set_xlim(zlim[0])
        ax_z.set_ylim(zlim[1])
        ax_z.set_aspect("equal", "box")
        ax_z.set_title(f"latent proposals | iteration {row.get('iteration')}")
        ax_z.set_xlabel("z1")
        ax_z.set_ylabel("z2")
        ax_z.legend(loc="best")
        ax_z.grid(True, alpha=0.18)

        pts = center_path(row.get("selected_decoded_path"))
        if target.shape[0] > 0:
            ax_path.plot(target[:, 0], target[:, 1], color="black", linestyle="--", linewidth=1.5, label="target")
        if pts.shape[0] > 0:
            ax_path.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=2.3, label="selected path")
            ax_path.scatter([pts[0, 0]], [pts[0, 1]], s=36, color="tab:blue")
        ax_path.set_xlim(xlim)
        ax_path.set_ylim(ylim)
        ax_path.set_aspect("equal", "box")
        ax_path.axis("off")
        ax_path.set_title("decoded selected path")
        ax_path.legend(loc="best")
        fig.suptitle(METHOD_LABELS.get(run.get("method"), run.get("method")))

    anim = FuncAnimation(fig, draw, frames=len(frame_rows), interval=1000 / max(1, fps))
    try:
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=120)
        ok = True
    except Exception:
        ok = False
    plt.close(fig)
    return ok


def mean_of_series(run, key):
    _x, y = series(run, key)
    return float(np.nanmean(y)) if np.any(np.isfinite(y)) else np.nan


def last_of_series(run, key):
    _x, y = series(run, key)
    finite = y[np.isfinite(y)]
    return float(finite[-1]) if finite.size else np.nan


def metric_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def rank_methods(method_summaries, key):
    vals = []
    for method, summary in method_summaries.items():
        mean = summary.get(key, {}).get("mean")
        vals.append((method, -np.inf if mean is None else float(mean)))
    vals.sort(key=lambda x: x[1], reverse=True)
    return [method for method, _val in vals]


def make_interpretation(method, summaries, rankings):
    final_rank = rankings["final_best_objective"].index(method) + 1
    novelty_rank = rankings["novelty"].index(method) + 1
    quality_rank = rankings["top_k_quality"].index(method) + 1
    adjusted_rank = rankings["adjusted_top_k_quality"].index(method) + 1
    return (
        f"Final-best rank {final_rank}; novelty rank {novelty_rank}; "
        f"top-k quality rank {quality_rank}; combined rank {adjusted_rank}."
    )


def write_summary(method_runs, summaries_dir):
    method_summaries = {}
    for method, runs in method_runs.items():
        method_summaries[method] = {
            "n_runs": len(runs),
            "final_best_objective": metric_stats([last_of_series(run, "best_so_far_objective") for run in runs]),
            "novelty": metric_stats([mean_of_series(run, "nearest_previous_selected_latent_distance") for run in runs]),
            "top_k_quality": metric_stats([mean_of_series(run, "mean_top_k_objective") for run in runs]),
            "top_k_diversity": metric_stats([mean_of_series(run, "top_k_latent_diversity") for run in runs]),
            "adjusted_top_k_quality": metric_stats([mean_of_series(run, "adjusted_top_k_objective") for run in runs]),
        }

    rankings = {
        "final_best_objective": rank_methods(method_summaries, "final_best_objective"),
        "novelty": rank_methods(method_summaries, "novelty"),
        "top_k_quality": rank_methods(method_summaries, "top_k_quality"),
        "top_k_diversity": rank_methods(method_summaries, "top_k_diversity"),
        "adjusted_top_k_quality": rank_methods(method_summaries, "adjusted_top_k_quality"),
    }

    for method in method_summaries:
        method_summaries[method]["interpretation"] = make_interpretation(method, method_summaries, rankings)

    payload = {
        "methods": method_summaries,
        "rankings": rankings,
    }

    json_path = os.path.join(summaries_dir, "summary_toy_diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(summaries_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "n_runs",
                "final_best_objective_mean",
                "novelty_mean",
                "top_k_quality_mean",
                "top_k_diversity_mean",
                "adjusted_top_k_quality_mean",
                "interpretation",
            ]
        )
        for method in sorted(method_summaries):
            s = method_summaries[method]
            writer.writerow(
                [
                    method,
                    s["n_runs"],
                    s["final_best_objective"]["mean"],
                    s["novelty"]["mean"],
                    s["top_k_quality"]["mean"],
                    s["top_k_diversity"]["mean"],
                    s["adjusted_top_k_quality"]["mean"],
                    s["interpretation"],
                ]
            )

    return json_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results/toy_diagnostics")
    ap.add_argument("--representative_iterations", type=int, nargs="*", default=[10, 50, 100])
    ap.add_argument("--make_gifs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gif_max_frames", type=int, default=80)
    ap.add_argument("--gif_seed_limit", type=int, default=1,
                    help="Number of seeds per method to animate; use 0 to animate all seeds.")
    args = ap.parse_args()

    runs = load_runs(args.results_root)
    if not runs:
        raise SystemExit(f"No metrics.json files found under {args.results_root!r}. Run experiments first.")

    method_runs = defaultdict(list)
    for run in runs:
        method_runs[str(run.get("method", "unknown"))].append(run)
    method_runs = dict(sorted(method_runs.items()))

    plots_dir = os.path.join(args.results_root, "plots")
    gifs_dir = os.path.join(args.results_root, "gifs")
    summaries_dir = os.path.join(args.results_root, "summaries")
    ensure_dir(plots_dir)
    ensure_dir(gifs_dir)
    ensure_dir(summaries_dir)

    plot_metric_grid(
        method_runs,
        MAIN_METRICS,
        os.path.join(plots_dir, "iteration_metrics.png"),
        "Toy diagnostics: iteration-wise method comparison",
    )
    plot_metric_grid(
        method_runs,
        RAW_TOPK_METRICS,
        os.path.join(plots_dir, "raw_top_k_components.png"),
        "Toy diagnostics: raw top-k components",
    )
    plot_latent_trajectories(method_runs, os.path.join(plots_dir, "latent_trajectories.png"))

    for method, run in first_run_by_method(method_runs).items():
        stem = f"{method}_seed_{int(run.get('seed', 0))}"
        plot_proposal_clouds(
            run,
            args.representative_iterations,
            os.path.join(plots_dir, f"proposal_clouds_{stem}.png"),
        )
        plot_decoded_paths(
            run,
            args.representative_iterations,
            os.path.join(plots_dir, f"decoded_paths_{stem}.png"),
        )

    made_gifs = []
    if args.make_gifs:
        for method, runs_for_method in method_runs.items():
            ordered = sorted(runs_for_method, key=lambda r: int(r.get("seed", 0)))
            if args.gif_seed_limit > 0:
                ordered = ordered[: args.gif_seed_limit]
            for run in ordered:
                stem = f"{method}_seed_{int(run.get('seed', 0))}"
                gif_path = os.path.join(gifs_dir, f"{stem}_diagnostics.gif")
                if save_run_gif(run, gif_path, max_frames=args.gif_max_frames):
                    made_gifs.append(gif_path)

    summary_json, summary_csv = write_summary(method_runs, summaries_dir)

    print("Loaded runs:", len(runs))
    print("Saved plots:", os.path.abspath(plots_dir))
    print("Saved gifs:", os.path.abspath(gifs_dir), f"({len(made_gifs)} created)")
    print("Saved summary JSON:", os.path.abspath(summary_json))
    print("Saved comparison table:", os.path.abspath(summary_csv))


if __name__ == "__main__":
    main()
