"""Plot COWBOYS-flow proposal sweep diagnostics.

This is a monitoring plotter for runs from
run_09_cowboys_flow_latent_proposal_sweep.sh. It treats each proposal
configuration as a separate curve, ordered from global to local proposals.
"""

import argparse
import csv
import glob
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

SHORT_LABELS = {
    "global_flow_only": "global",
    "mostly_global_tight": "g-tight",
    "mostly_global_wide": "g-wide",
    "sharp_global": "sharp g",
    "soft_global": "soft g",
    "balanced_tight": "bal tight",
    "balanced_base": "bal base",
    "balanced_wide": "bal wide",
    "mostly_local_tight": "l-tight",
    "mostly_local_wide": "l-wide",
    "local_only_tight": "local tight",
    "local_only_wide": "local wide",
}

REPRESENTATIVE_LABELS = [
    "global_flow_only",
    "soft_global",
    "balanced_base",
    "mostly_local_tight",
    "local_only_tight",
]

LABEL_RANK = {
    "global_flow_only": 0,
    "mostly_global_tight": 1,
    "mostly_global_wide": 2,
    "sharp_global": 3,
    "soft_global": 4,
    "balanced_tight": 5,
    "balanced_base": 6,
    "balanced_wide": 7,
    "mostly_local_tight": 8,
    "mostly_local_wide": 9,
    "local_only_tight": 10,
    "local_only_wide": 11,
}

MARKERS = {
    "tight": "o",
    "wide": "^",
    "base": "s",
    "only": "D",
    "sharp": "*",
    "soft": "X",
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


def infer_tag_from_plotroot(plotroot):
    if not plotroot:
        return None
    parts = re.split(r"[\\/]+", str(plotroot))
    try:
        idx = parts.index("cowboys_flow_latent_proposal_sweep")
    except ValueError:
        return None
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return None


def infer_label_from_tag(tag):
    tag = str(tag or "")
    for marker in ("_pi_bt", "_ei_bt"):
        if marker in tag:
            return tag.split(marker, 1)[0]
    match = re.match(r"(.+?)_(?:pi|ei)_", tag)
    return match.group(1) if match else tag


def parse_float_from_tag(tag, name):
    match = re.search(rf"(?:^|_){re.escape(name)}(-?\d+(?:\.\d+)?)", str(tag or ""))
    if match:
        return finite_float(match.group(1))
    return np.nan


def run_identity_from_config(config, fallback_tag=None):
    config = config or {}
    tag = infer_tag_from_plotroot(config.get("plotroot")) or fallback_tag or ""
    label_key = infer_label_from_tag(tag)
    lam = finite_float(config.get("lambda_local"))
    if not np.isfinite(lam):
        lam = parse_float_from_tag(tag, "lam")
    sigma = finite_float(config.get("sigma_local"))
    if not np.isfinite(sigma):
        sigma = parse_float_from_tag(tag, "sig")
    beta = finite_float(config.get("beta_tilt"))
    tau = finite_float(config.get("tau_temp"))
    return {
        "tag": tag or label_key,
        "label_key": label_key,
        "lambda_local": lam,
        "sigma_local": sigma,
        "beta_tilt": beta,
        "tau_temp": tau,
    }


def short_label(info):
    base = SHORT_LABELS.get(info["label_key"], info["label_key"].replace("_", " "))
    lam = info.get("lambda_local", np.nan)
    if np.isfinite(lam):
        return f"{base} lam{lam:.2g}"
    return base


def sort_key(info):
    lam = info.get("lambda_local", np.nan)
    lam_sort = lam if np.isfinite(lam) else 99.0
    return (lam_sort, LABEL_RANK.get(info.get("label_key"), 999), info.get("tag", ""))


def color_for_lambda(lam):
    if not np.isfinite(lam):
        lam = 0.5
    lam = float(np.clip(lam, 0.0, 1.0))
    # Low lambda = more global = darker/harder; high lambda = more local = softer.
    return plt.cm.Reds(0.90 - 0.45 * lam)


def style_for_label(label_key):
    marker = None
    for key, value in MARKERS.items():
        if key in label_key:
            marker = value
            break
    if marker is None:
        marker = "o"
    if "wide" in label_key:
        linestyle = "--"
    elif "sharp" in label_key:
        linestyle = ":"
    elif "soft" in label_key:
        linestyle = "-."
    else:
        linestyle = "-"
    return linestyle, marker


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


def load_json_runs(diagnostics_root):
    pattern = os.path.join(
        diagnostics_root,
        "raw",
        "cowboys_flow_latent",
        "**",
        "metrics.json",
    )
    runs = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(windows_long_path(path), "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = payload.get("config", {})
        tag = infer_tag_from_plotroot(config.get("plotroot"))
        if "cowboys_flow_latent_proposal_sweep" not in str(config.get("plotroot", "")) and tag is None:
            continue
        payload["_source"] = "diagnostics_json"
        payload["_path"] = path
        payload["_identity"] = run_identity_from_config(config, fallback_tag=tag)
        runs.append(payload)
    return runs


def load_csv_fallback_runs(runs_root):
    pattern = os.path.join(
        runs_root,
        "cowboys_flow_latent_proposal_sweep",
        "*",
        "seed_*",
        "metrics.csv",
    )
    runs = []
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        config_path = os.path.join(run_dir, "run_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(windows_long_path(config_path), "r", encoding="utf-8") as f:
                config = json.load(f)
        rows = []
        with open(windows_long_path(path), "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            continue
        it = [int(float(row["iter"])) for row in rows if row.get("iter")]
        best = [finite_float(row.get("best_y")) for row in rows]
        payload = {
            "method": "cowboys_flow_latent",
            "seed": int(re.search(r"seed_(\d+)", run_dir).group(1)) if re.search(r"seed_(\d+)", run_dir) else 0,
            "config": config,
            "metric_series": {
                "bo_iteration": it,
                "best_so_far_objective": best,
                "nearest_previous_selected_latent_distance": [],
                "adjusted_top_k_objective": [],
            },
            "_source": "metrics_csv",
            "_path": path,
            "_identity": run_identity_from_config(config, fallback_tag=os.path.basename(os.path.dirname(run_dir))),
        }
        runs.append(payload)
    return runs


def group_runs(json_runs, csv_runs):
    grouped = defaultdict(list)
    have_full = set()
    info_by_tag = {}

    for run in json_runs:
        info = run["_identity"]
        tag = info["tag"]
        seed = int(run.get("seed", 0))
        grouped[tag].append(run)
        have_full.add((tag, seed))
        info_by_tag[tag] = info

    for run in csv_runs:
        info = run["_identity"]
        tag = info["tag"]
        seed = int(run.get("seed", 0))
        if (tag, seed) in have_full:
            continue
        grouped[tag].append(run)
        info_by_tag[tag] = info

    return grouped, info_by_tag


def plot_sweep(
    grouped,
    info_by_tag,
    save_path,
    bin_size=10,
    robust_y=True,
    title_note="",
    tight_diagnostic_y=False,
    diagnostic_low_q=0.10,
    diagnostic_high_q=0.90,
    no_main_title=False,
    thesis=False,
):
    ordered_tags = sorted(grouped, key=lambda tag: sort_key(info_by_tag[tag]))
    if not ordered_tags:
        raise SystemExit("No COWBOYS flow proposal-sweep diagnostics found.")

    fig_size = (18.2, 5.2) if thesis else (19.0, 5.2)
    title_fs = 16 if thesis else None
    label_fs = 13 if thesis else None
    tick_fs = 11 if thesis else None
    legend_fs = 14 if thesis else 8.5
    line_width = 2.5 if thesis else 2.25
    marker_size = 5.0 if thesis else 4.3

    fig, axes = plt.subplots(1, 3, figsize=fig_size, squeeze=False)
    axes = axes[0]
    legend_handles = []
    legend_labels = []

    for ax, (metric_key, title) in zip(axes, METRICS):
        axis_values = []
        missing = []
        for tag in ordered_tags:
            info = info_by_tag[tag]
            runs = grouped[tag]
            x, mean, spread, counts = aggregate_runs(runs, metric_key, bin_size=bin_size)
            if x.size == 0 or np.all(~np.isfinite(mean)):
                if metric_key != "best_so_far_objective":
                    missing.append(short_label(info))
                continue

            color = color_for_lambda(info.get("lambda_local", np.nan))
            linestyle, marker = style_for_label(info.get("label_key", ""))
            label = short_label(info)
            markevery = max(1, len(x) // 12)
            line, = ax.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=line_width,
                marker=marker,
                markersize=marker_size,
                markevery=markevery,
                label=label,
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
                legend_labels.append(label)

        if robust_y:
            low_q, high_q = 0.03, 0.97
            if tight_diagnostic_y and metric_key != "best_so_far_objective":
                low_q, high_q = diagnostic_low_q, diagnostic_high_q
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
                        fontsize=9 if thesis else 8,
                        color="0.35",
                    )
        if missing:
            missing_text = ", ".join(missing[:6])
            if len(missing) > 6:
                missing_text += f", +{len(missing) - 6}"
            ax.text(
                0.01,
                0.98,
                "missing full diagnostics: " + missing_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9 if thesis else 8,
                color="0.35",
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.85", linewidth=0.5),
            )
        ax.set_title(title, fontsize=title_fs, pad=10 if thesis else None)
        ax.set_xlabel("BO iteration", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("metric value", fontsize=label_fs)

    if legend_handles:
        legend_y = 0.035 if thesis else -0.015
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, len(legend_labels)),
            frameon=True,
            fontsize=legend_fs,
            bbox_to_anchor=(0.5, legend_y),
        )

    if not no_main_title:
        title = "COWBOYS Flow proposal sweep diagnostics"
        if title_note:
            title += f" ({title_note})"
        subtitle = "global-to-local proposal sweep; darker red = more global, softer red = more local"
        if bin_size > 1:
            subtitle += f"; plotted in {bin_size}-iteration bins"
        if tight_diagnostic_y:
            subtitle += "; tighter diagnostic y-scale"
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
        fig.tight_layout(rect=(0.0, 0.24 if thesis else 0.13, 1.0, 0.90))
    else:
        fig.tight_layout(rect=(0.0, 0.30 if thesis else 0.13, 1.0, 0.98))
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_sweeps")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs")
    ap.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    ap.add_argument("--preset", type=str, default="all", choices=["all", "representative"],
                    help="Use all configs or a small global-to-local representative subset.")
    ap.add_argument("--include_labels", type=str, default="",
                    help="Comma-separated config labels to plot, e.g. global_flow_only,soft_global.")
    ap.add_argument("--bin_size", type=int, default=10)
    ap.add_argument("--no_csv_fallback", action="store_true",
                    help="Use only finalized diagnostics JSON, not partial metrics.csv best-so-far traces.")
    ap.add_argument("--no_robust_y", action="store_true")
    ap.add_argument("--tight_diagnostic_y", action="store_true",
                    help="Use tighter percentile y-limits for nearest-distance and adjusted-top-k panels.")
    ap.add_argument("--diagnostic_low_q", type=float, default=0.10)
    ap.add_argument("--diagnostic_high_q", type=float, default=0.90)
    ap.add_argument("--no_main_title", action="store_true")
    ap.add_argument("--thesis", action="store_true",
                    help="Use larger subplot titles, axis labels, ticks, and legend text for thesis figures.")
    args = ap.parse_args()

    json_runs = load_json_runs(args.diagnostics_root)
    csv_runs = [] if args.no_csv_fallback else load_csv_fallback_runs(args.runs_root)
    grouped, info_by_tag = group_runs(json_runs, csv_runs)

    selected_labels = []
    title_note = ""
    if args.include_labels.strip():
        selected_labels = [part.strip() for part in args.include_labels.split(",") if part.strip()]
        title_note = "selected"
    elif args.preset == "representative":
        selected_labels = REPRESENTATIVE_LABELS
        title_note = "representative subset"

    if selected_labels:
        selected = set(selected_labels)
        grouped = {
            tag: runs
            for tag, runs in grouped.items()
            if info_by_tag[tag].get("label_key") in selected
        }
        info_by_tag = {tag: info for tag, info in info_by_tag.items() if tag in grouped}

    if args.save_path is None:
        filename = "proposal_sweep_diagnostics.png"
        if args.preset != "all" or args.include_labels.strip():
            filename = f"proposal_sweep_{args.preset}.png" if not args.include_labels.strip() else "proposal_sweep_selected.png"
        args.save_path = os.path.join(
            args.runs_root,
            "cowboys_flow_latent_proposal_sweep",
            "plots",
            filename,
        )

    plot_sweep(
        grouped=grouped,
        info_by_tag=info_by_tag,
        save_path=args.save_path,
        bin_size=args.bin_size,
        robust_y=not args.no_robust_y,
        title_note=title_note,
        tight_diagnostic_y=args.tight_diagnostic_y,
        diagnostic_low_q=args.diagnostic_low_q,
        diagnostic_high_q=args.diagnostic_high_q,
        no_main_title=args.no_main_title,
        thesis=args.thesis,
    )

    n_runs = sum(len(runs) for runs in grouped.values())
    print(f"Loaded {len(grouped)} configs / {n_runs} run traces.")
    print("Saved:", os.path.abspath(args.save_path))


if __name__ == "__main__":
    main()
