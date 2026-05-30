"""Compare DGBO variants with matched guidance configurations.

This plotter is intended for the dgbo_variants folder where normal DGBO,
distillation, and REINFORCE runs share guidance settings. Curves are colored by
method, while marker/line style identifies the guidance configuration.
"""

import argparse
import json
import os
import re
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

METHOD_LABELS = {
    "dgbo_latent_diffusion": "DGBO",
    "dgbo": "DGBO",
    "dgbo_dist": "DGBO Distill",
    "dgbo_distillation": "DGBO Distill",
    "dgbo_latent_diffusion_reinforce": "DGBO REINFORCE",
    "dgbo_reinforce": "DGBO REINFORCE",
}

METHOD_COLORS = {
    "DGBO": "#1f77b4",
    "DGBO Distill": "#d62728",
    "DGBO REINFORCE": "#2ca02c",
}

CONFIG_LABELS = {
    "unguided": "g0 none",
    "soft_clipped": "g0.5 soft",
    "base_mid_clip": "g1 base",
    "strong_default": "g2 strong",
    "very_strong_high_tau": "g4 high-tau",
}

CONFIG_STYLES = {
    "unguided": (":", "o"),
    "soft_clipped": ("--", "s"),
    "base_mid_clip": ("-", "^"),
    "strong_default": ("-.", "D"),
    "very_strong_high_tau": (":", "P"),
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


def canonical_run_id(run_id):
    text = str(run_id or "")
    for prefix in ("distill_", "real_", "reinf_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("stro_default", "strong_default")
    return text


def parse_config_key(run_id, config):
    run_id = canonical_run_id(run_id)
    match = re.match(
        r"^(?P<label>.+)_tau(?P<tau>-?\d+(?:\.\d+)?)_gs(?P<gs>-?\d+(?:\.\d+)?)_clip(?P<clip>-?\d+(?:\.\d+)?)",
        run_id,
    )
    if match:
        label = match.group("label")
        gs = finite_float(match.group("gs"))
        tau = finite_float(match.group("tau"))
        clip = finite_float(match.group("clip"))
    else:
        label = run_id
        gs = finite_float(config.get("guidance_scale"))
        tau = finite_float(config.get("tau_guidance"))
        clip = finite_float(config.get("clip_guidance"))
    return label, gs, tau, clip


def method_label(method):
    return METHOD_LABELS.get(str(method), str(method).replace("_", " "))


def short_config_label(label, gs, tau, clip):
    if label in CONFIG_LABELS:
        return CONFIG_LABELS[label]
    if np.isfinite(gs):
        return f"g{gs:g} tau{tau:g} clip{clip:g}"
    return str(label).replace("_", " ")


def config_sort_key(item):
    label, gs, tau, clip = item
    order = {
        "unguided": 0,
        "soft_clipped": 1,
        "base_mid_clip": 2,
        "strong_default": 3,
        "very_strong_high_tau": 4,
    }
    gs_sort = gs if np.isfinite(gs) else 99.0
    tau_sort = tau if np.isfinite(tau) else 99.0
    clip_sort = clip if np.isfinite(clip) else 99.0
    return (gs_sort, tau_sort, clip_sort, order.get(label, 999), label)


def reduce_bin(values, reducer):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if reducer == "last":
        return float(vals[-1])
    return float(np.mean(vals))


def bin_series(x, y, key, bin_size):
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
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


def aggregate_runs(runs, key, bin_size):
    series = []
    for run in runs:
        metric = run.get("metric_series", {})
        x = np.asarray(metric.get("bo_iteration", []), dtype=np.int64)
        y = np.asarray([finite_float(v) for v in metric.get(key, [])], dtype=np.float64)
        n = min(x.size, y.size)
        if n == 0:
            continue
        series.append(bin_series(x[:n], y[:n], key, bin_size))

    all_iters = sorted({int(i) for x, _ in series for i in x})
    if not all_iters:
        return np.zeros((0,), dtype=int), np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=int)

    mat = np.full((len(series), len(all_iters)), np.nan, dtype=np.float64)
    pos = {it: j for j, it in enumerate(all_iters)}
    for r, (x, y) in enumerate(series):
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
        spread[j] = float(np.std(vals) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
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


def load_runs(diagnostics_root, method_dir):
    root = os.path.join(diagnostics_root, "raw", method_dir)
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics.json" not in filenames:
            continue
        path = os.path.join(dirpath, "metrics.json")
        with open(windows_long_path(path), "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = payload.get("config", {})
        run_id = config.get("diagnostics_run_id") or os.path.basename(os.path.dirname(os.path.dirname(path)))
        label, gs, tau, clip = parse_config_key(run_id, config)
        payload["_variant"] = {
            "method": method_label(payload.get("method")),
            "config_key": (label, gs, tau, clip),
            "config_label": short_config_label(label, gs, tau, clip),
        }
        runs.append(payload)
    return runs


def matched_groups(runs, exclude_labels=None, include_labels=None):
    exclude_labels = set(exclude_labels or [])
    include_labels = set(include_labels or [])
    required_methods = {"DGBO", "DGBO Distill", "DGBO REINFORCE"}
    grouped = defaultdict(list)
    methods_by_config = defaultdict(set)
    for run in runs:
        variant = run["_variant"]
        label = variant["config_key"][0]
        if label in exclude_labels:
            continue
        if include_labels and label not in include_labels:
            continue
        key = (variant["method"], variant["config_key"])
        grouped[key].append(run)
        methods_by_config[variant["config_key"]].add(variant["method"])

    matched_configs = {
        config_key
        for config_key, methods in methods_by_config.items()
        if required_methods.issubset(methods)
    }
    filtered = {
        key: value
        for key, value in grouped.items()
        if key[1] in matched_configs
    }
    return filtered, sorted(matched_configs, key=config_sort_key)


def plot_comparison(
    grouped,
    ordered_configs,
    save_path,
    bin_size=10,
    tight_diagnostic_y=False,
    thesis=False,
    no_main_title=False,
):
    if not grouped:
        raise SystemExit("No matched DGBO/DGBO-distillation/DGBO-REINFORCE diagnostics found.")

    if thesis:
        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.labelsize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
            }
        )
    title_fs = 17 if thesis else None
    label_fs = 15 if thesis else None
    legend_fs = 12 if thesis else 8.2
    line_width = 2.7 if thesis else 2.25
    marker_size = 5.0 if thesis else 4.2

    fig_height = 5.9 if thesis else 5.2
    fig, axes = plt.subplots(1, 3, figsize=(19.0, fig_height), squeeze=False)
    axes = axes[0]
    legend_handles = []
    legend_labels = []
    legend_keys = []

    methods = ["DGBO", "DGBO Distill", "DGBO REINFORCE"]
    method_pos = {method: i for i, method in enumerate(methods)}
    config_pos = {config_key: i for i, config_key in enumerate(ordered_configs)}
    for ax, (metric_key, title) in zip(axes, METRICS):
        axis_values = []
        for config_key in ordered_configs:
            label, gs, tau, clip = config_key
            linestyle, marker = CONFIG_STYLES.get(label, ("-", "o"))
            config_label = short_config_label(label, gs, tau, clip)
            for method in methods:
                runs = grouped.get((method, config_key), [])
                if not runs:
                    continue
                x, mean, spread, counts = aggregate_runs(runs, metric_key, bin_size=bin_size)
                if x.size == 0 or np.all(~np.isfinite(mean)):
                    continue

                color = METHOD_COLORS.get(method, "0.25")
                line_label = f"{method}: {config_label}"
                line, = ax.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=line_width,
                    marker=marker,
                    markersize=marker_size,
                    markevery=max(1, len(x) // 12),
                    label=line_label,
                )
                axis_values.extend(mean[np.isfinite(mean)].tolist())

                ok = np.isfinite(mean) & np.isfinite(spread) & (counts > 1)
                if np.any(ok):
                    lower = mean[ok] - spread[ok]
                    upper = mean[ok] + spread[ok]
                    axis_values.extend(lower.tolist())
                    axis_values.extend(upper.tolist())
                    ax.fill_between(x[ok], lower, upper, color=color, alpha=0.10, linewidth=0)

                if metric_key == METRICS[0][0]:
                    legend_handles.append(line)
                    legend_labels.append(line_label)
                    legend_keys.append((method_pos.get(method, 99), config_pos.get(config_key, 99)))

        low_q, high_q = 0.03, 0.97
        if tight_diagnostic_y and metric_key != "best_so_far_objective":
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
                    fontsize=8,
                    color="0.35",
                )
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("BO iteration", fontsize=label_fs)
        if thesis:
            ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("metric value", fontsize=label_fs)

    if legend_handles:
        order = sorted(range(len(legend_handles)), key=lambda i: legend_keys[i])
        fig.legend(
            [legend_handles[i] for i in order],
            [legend_labels[i] for i in order],
            loc="lower center",
            ncol=3,
            frameon=True,
            fontsize=legend_fs,
            bbox_to_anchor=(0.5, -0.055 if thesis else -0.075),
            handlelength=2.5,
            columnspacing=1.35,
        )

    if not no_main_title:
        subtitle = "color = method; line style/marker = matched guidance configuration"
        if bin_size > 1:
            subtitle += f"; plotted in {bin_size}-iteration bins"
        if tight_diagnostic_y:
            subtitle += "; tighter diagnostic y-scale"
        fig.suptitle(f"DGBO variant comparison\n{subtitle}", fontsize=15 if thesis else 13)
        fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.90))
    else:
        fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.98))
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(
        windows_long_path(save_path),
        dpi=300 if thesis else 190,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_dgbo_variants")
    ap.add_argument("--method_dir", type=str, default="dgbo_latent_distillation")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs")
    ap.add_argument("--run_group", type=str, default="dgbo_variants")
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--bin_size", type=int, default=10)
    ap.add_argument("--tight_diagnostic_y", action="store_true")
    ap.add_argument("--thesis", action="store_true", help="Use larger thesis-style text and export settings.")
    ap.add_argument("--no_main_title", action="store_true", help="Omit the figure-level title/subtitle.")
    ap.add_argument(
        "--include_labels",
        type=str,
        default="",
        help="Comma-separated guidance labels to keep, e.g. soft_clipped,base_mid_clip.",
    )
    ap.add_argument(
        "--exclude_labels",
        type=str,
        default="",
        help="Comma-separated guidance labels to drop, e.g. unguided.",
    )
    args = ap.parse_args()

    runs = load_runs(args.diagnostics_root, args.method_dir)
    exclude_labels = [part.strip() for part in args.exclude_labels.split(",") if part.strip()]
    include_labels = [part.strip() for part in args.include_labels.split(",") if part.strip()]
    grouped, ordered_configs = matched_groups(
        runs,
        exclude_labels=exclude_labels,
        include_labels=include_labels,
    )
    if args.save_path is None:
        name = "dgbo_variants_method_comparison.png"
        if args.tight_diagnostic_y:
            name = "dgbo_variants_method_comparison_tight.png"
        args.save_path = os.path.join(args.runs_root, args.run_group, "plots", name)

    plot_comparison(
        grouped=grouped,
        ordered_configs=ordered_configs,
        save_path=args.save_path,
        bin_size=args.bin_size,
        tight_diagnostic_y=args.tight_diagnostic_y,
        thesis=args.thesis,
        no_main_title=args.no_main_title,
    )

    n_runs = sum(len(items) for items in grouped.values())
    print(f"Loaded {n_runs} matched run traces across {len(ordered_configs)} configs.")
    print("Saved:", os.path.abspath(args.save_path))


if __name__ == "__main__":
    main()
