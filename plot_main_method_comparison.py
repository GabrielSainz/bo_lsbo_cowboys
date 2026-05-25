"""Thesis-style method comparison for the main toy benchmark runs."""

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("best_so_far_objective", "Best-so-far objective"),
    ("nearest_previous_selected_latent_distance", "Nearest previous selected latent distance"),
    ("adjusted_top_k_objective", "Adjusted top-k objective"),
]

METHOD_ORDER = [
    "lsbo",
    "cowboys",
    "dgbo_latent_diffusion",
    "cowboys_flow_latent",
]

METHOD_LABELS = {
    "lsbo": "LSBO",
    "cowboys": "COWBOYS",
    "dgbo_latent_diffusion": "DGBO",
    "cowboys_flow_latent": "NFlow",
}

METHOD_COLORS = {
    "lsbo": "#1f77b4",
    "cowboys": "#ff7f0e",
    "dgbo_latent_diffusion": "#9467bd",
    "cowboys_flow_latent": "#d62728",
}

METHOD_MARKERS = {
    "lsbo": "o",
    "cowboys": "s",
    "dgbo_latent_diffusion": "^",
    "cowboys_flow_latent": "P",
}

METHOD_LINESTYLES = {
    "lsbo": "-",
    "cowboys": "-",
    "dgbo_latent_diffusion": "--",
    "cowboys_flow_latent": "-",
}


def windows_long_path(path):
    path = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def ensure_dir(path):
    os.makedirs(windows_long_path(path), exist_ok=True)


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_runs(diagnostics_root):
    root = os.path.join(diagnostics_root, "raw")
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics.json" not in filenames:
            continue
        path = os.path.join(dirpath, "metrics.json")
        with open(windows_long_path(path), "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = path
        runs.append(payload)
    return runs


def filter_nflow_runs(runs, nflow_run_id_contains=None):
    token = str(nflow_run_id_contains or "").strip()
    if not token:
        return runs

    filtered = []
    dropped = 0
    for run in runs:
        method = str(run.get("method", "unknown"))
        if method != "cowboys_flow_latent":
            filtered.append(run)
            continue
        config = run.get("config", {})
        run_id = str(config.get("diagnostics_run_id") or "")
        path = str(run.get("_path") or "")
        if token in run_id or token in path:
            filtered.append(run)
        else:
            dropped += 1
    print(f"NFlow selector '{token}': dropped {dropped} non-matching NFlow runs.")
    return filtered


def ordered_methods(method_runs):
    known = [method for method in METHOD_ORDER if method in method_runs]
    extra = sorted(method for method in method_runs if method not in METHOD_ORDER)
    return known + extra


def reduce_bin(values, reducer):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if reducer == "last":
        return float(vals[-1])
    return float(np.mean(vals))


def series(run, key, bin_size):
    metric = run.get("metric_series", {})
    x_raw = metric.get("bo_iteration", [])
    y_raw = metric.get(key, [])
    n = min(len(x_raw), len(y_raw))
    x = np.asarray(x_raw[:n], dtype=np.int64)
    y = np.asarray([finite_float(v) for v in y_raw[:n]], dtype=np.float64)
    if bin_size <= 1 or x.size == 0:
        return x, y

    reducer = "last" if key == "best_so_far_objective" else "mean"
    bx = []
    by = []
    max_it = int(np.nanmax(x))
    for start in range(1, max_it + 1, int(bin_size)):
        end = start + int(bin_size) - 1
        mask = (x >= start) & (x <= end)
        if not np.any(mask):
            continue
        bx.append(int(min(end, max_it)))
        by.append(reduce_bin(y[mask], reducer))
    return np.asarray(bx, dtype=np.int64), np.asarray(by, dtype=np.float64)


def aggregate_series(runs, key, bin_size, variability):
    all_iters = sorted({int(i) for run in runs for i in series(run, key, bin_size)[0]})
    if not all_iters:
        return np.zeros((0,), dtype=int), np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=int)

    mat = np.full((len(runs), len(all_iters)), np.nan, dtype=np.float64)
    pos = {it: j for j, it in enumerate(all_iters)}
    for r, run in enumerate(runs):
        x, y = series(run, key, bin_size)
        for it, val in zip(x, y):
            mat[r, pos[int(it)]] = val

    mean = np.full(len(all_iters), np.nan, dtype=np.float64)
    spread = np.full(len(all_iters), np.nan, dtype=np.float64)
    counts = np.sum(np.isfinite(mat), axis=0).astype(int)
    for j in range(len(all_iters)):
        vals = mat[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean[j] = float(np.mean(vals))
        if vals.size > 1:
            std = float(np.std(vals))
            spread[j] = std if variability == "std" else std / np.sqrt(vals.size)
        else:
            spread[j] = 0.0
    return np.asarray(all_iters, dtype=int), mean, spread, counts


def robust_limits(values, low_q=0.03, high_q=0.97):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, False
    lo = float(np.quantile(vals, low_q))
    hi = float(np.quantile(vals, high_q))
    if abs(hi - lo) < 1e-12:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    span = max(hi - lo, 1e-6)
    clipped = bool(np.nanmin(vals) < lo or np.nanmax(vals) > hi)
    return (lo - 0.10 * span, hi + 0.10 * span), clipped


def plot_comparison(
    method_runs,
    save_path,
    bin_size=10,
    variability="std",
    robust_y=True,
    tight_diagnostic_y=False,
    no_main_title=True,
):
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.2), squeeze=False)
    axes = axes[0]
    legend_handles = []
    legend_labels = []

    for ax, (key, title) in zip(axes, METRICS):
        axis_values = []
        for method in ordered_methods(method_runs):
            runs = method_runs[method]
            x, mean, spread, counts = aggregate_series(runs, key, bin_size, variability)
            if x.size == 0 or np.all(~np.isfinite(mean)):
                continue

            color = METHOD_COLORS.get(method, "0.2")
            label = METHOD_LABELS.get(method, method)
            line, = ax.plot(
                x,
                mean,
                color=color,
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                linewidth=2.6,
                marker=METHOD_MARKERS.get(method, "o"),
                markersize=5.2,
                markevery=max(1, len(x) // 12),
                label=label,
            )
            axis_values.extend(mean[np.isfinite(mean)].tolist())

            ok = np.isfinite(mean) & np.isfinite(spread) & (counts > 1)
            if np.any(ok):
                lower = mean[ok] - spread[ok]
                upper = mean[ok] + spread[ok]
                axis_values.extend(lower.tolist())
                axis_values.extend(upper.tolist())
                ax.fill_between(x[ok], lower, upper, color=color, alpha=0.14, linewidth=0)

            if key == METRICS[0][0]:
                legend_handles.append(line)
                legend_labels.append(label)

        if robust_y:
            low_q, high_q = 0.03, 0.97
            if tight_diagnostic_y and key != "best_so_far_objective":
                low_q, high_q = 0.10, 0.90
            ylim, clipped = robust_limits(axis_values, low_q=low_q, high_q=high_q)
            if ylim is not None:
                ax.set_ylim(ylim)
                if clipped:
                    ax.text(
                        0.99,
                        0.02,
                        "robust y-scale",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9,
                        color="0.35",
                    )

        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel("BO iteration", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("metric value", fontsize=13)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=True,
            fontsize=14,
            bbox_to_anchor=(0.5, 0.035),
        )

    if not no_main_title:
        band = "standard deviation" if variability == "std" else "standard error"
        subtitle = f"shaded band: {band} across seeds"
        if bin_size > 1:
            subtitle += f"; plotted in {bin_size}-iteration bins"
        fig.suptitle(f"Toy method comparison\n{subtitle}", fontsize=13)
        fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.90))
    else:
        fig.tight_layout(rect=(0.0, 0.30, 1.0, 0.98))

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_main_analysis")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs/main_comparison")
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--bin_size", type=int, default=10)
    ap.add_argument("--variability", type=str, default="std", choices=["std", "sem"])
    ap.add_argument("--tight_diagnostic_y", action="store_true")
    ap.add_argument("--no_main_title", action="store_true", default=True)
    ap.add_argument(
        "--nflow_run_id_contains",
        type=str,
        default="",
        help="Keep only NFlow diagnostics whose run id or path contains this text.",
    )
    args = ap.parse_args()

    runs = filter_nflow_runs(load_runs(args.diagnostics_root), args.nflow_run_id_contains)
    method_runs = defaultdict(list)
    for run in runs:
        method_runs[str(run.get("method", "unknown"))].append(run)
    method_runs = dict(method_runs)

    if args.save_path is None:
        name = "main_method_comparison_thesis_sd.pdf"
        if args.tight_diagnostic_y:
            name = "main_method_comparison_thesis_sd_zoom.pdf"
        args.save_path = os.path.join(args.runs_root, "plots", name)

    plot_comparison(
        method_runs,
        args.save_path,
        bin_size=args.bin_size,
        variability=args.variability,
        tight_diagnostic_y=args.tight_diagnostic_y,
        no_main_title=args.no_main_title,
    )

    print(f"Loaded {sum(len(v) for v in method_runs.values())} runs across {len(method_runs)} methods.")
    print("Saved:", os.path.abspath(args.save_path))


if __name__ == "__main__":
    main()
