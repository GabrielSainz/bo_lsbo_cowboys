"""Plot DGBO guidance sweep diagnostics.

This makes the same 3-panel monitoring plot used for the COWBOYS-flow
proposal sweep, but treats each DGBO guidance configuration as its own curve.
Use the all preset to inspect every completed run, and the representative
preset for one curve at each broad guidance-strength level.
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

SHORT_LABELS = {
    "unguided": "g0 none",
    "soft_low_tau": "g0.5 low-tau",
    "soft_clipped": "g0.5 soft",
    "base_low_clip": "g1 low-clip",
    "base_mid_clip": "g1 base",
    "base_unclipped": "g1 no-clip",
    "global_soft_clip": "g1 global",
    "glob_soft_clip": "g1 global",
    "strong_tight": "g2 tight",
    "strong_default": "g2 strong",
    "stro_default": "g2 strong",
    "strong_high_tau": "g2 high-tau",
    "stro_high_tau": "g2 high-tau",
    "very_strong_clip": "g4 v-strong",
    "ve_stro_clip": "g4 v-strong",
    "very_strong_high_tau": "g4 high-tau",
    "ve_str_hi_tau": "g4 high-tau",
    "extreme_clip": "g6 extreme",
    "extr_clip": "g6 extreme",
    "extreme_high_tau": "g6 high-tau",
    "extr_high_tau": "g6 high-tau",
    "ultra_clip": "g8 ultra",
    "ultra_high_tau": "g8 high-tau",
    "ult_hi_tau": "g8 high-tau",
}

REPRESENTATIVE_LABELS = [
    "unguided",
    "soft_clipped",
    "base_mid_clip",
    "strong_default",
    "very_strong_clip",
    "ultra_clip",
]

LABEL_RANK = {label: i for i, label in enumerate([
    "unguided",
    "soft_low_tau",
    "soft_clipped",
    "global_soft_clip",
    "glob_soft_clip",
    "base_low_clip",
    "base_mid_clip",
    "base_unclipped",
    "strong_tight",
    "strong_default",
    "stro_default",
    "strong_high_tau",
    "stro_high_tau",
    "very_strong_clip",
    "ve_stro_clip",
    "very_strong_high_tau",
    "ve_str_hi_tau",
    "extreme_clip",
    "extr_clip",
    "extreme_high_tau",
    "extr_high_tau",
    "ultra_clip",
    "ultra_high_tau",
    "ult_hi_tau",
])}


def ensure_dir(path):
    os.makedirs(windows_long_path(path), exist_ok=True)


def windows_long_path(path):
    path = os.path.abspath(path)
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def parse_tag(tag):
    tag = str(tag or "")
    match = re.match(
        r"^(?P<label>.+)_tau(?P<tau>-?\d+(?:\.\d+)?)_gs(?P<gs>-?\d+(?:\.\d+)?)_clip(?P<clip>-?\d+(?:\.\d+)?)",
        tag,
    )
    if not match:
        return tag, np.nan, np.nan, np.nan
    return (
        match.group("label"),
        finite_float(match.group("gs")),
        finite_float(match.group("tau")),
        finite_float(match.group("clip")),
    )


def tag_from_plotroot(plotroot):
    if not plotroot:
        return ""
    parts = re.split(r"[\\/]+", str(plotroot))
    for group in ("dgbo_latent_diffusion_sweep", "dgbo_guidance_sweep"):
        if group in parts:
            idx = parts.index(group)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def run_identity_from_config(config, fallback_tag=""):
    config = config or {}
    tag = (
        str(config.get("diagnostics_run_id") or "")
        or tag_from_plotroot(config.get("plotroot"))
        or fallback_tag
    )
    label_key, gs_tag, tau_tag, clip_tag = parse_tag(tag)
    guidance_scale = finite_float(config.get("guidance_scale"))
    tau_guidance = finite_float(config.get("tau_guidance"))
    clip_guidance = finite_float(config.get("clip_guidance"))
    if not np.isfinite(guidance_scale):
        guidance_scale = gs_tag
    if not np.isfinite(tau_guidance):
        tau_guidance = tau_tag
    if not np.isfinite(clip_guidance):
        clip_guidance = clip_tag
    return {
        "tag": tag or label_key,
        "label_key": label_key,
        "guidance_scale": guidance_scale,
        "tau_guidance": tau_guidance,
        "clip_guidance": clip_guidance,
    }


def short_label(info):
    label_key = info.get("label_key", "")
    base = SHORT_LABELS.get(label_key, label_key.replace("_", " "))
    gs = info.get("guidance_scale", np.nan)
    if np.isfinite(gs) and "g" not in base.split()[0]:
        return f"{base} g{gs:g}"
    return base


def sort_key(info):
    gs = info.get("guidance_scale", np.nan)
    tau = info.get("tau_guidance", np.nan)
    clip = info.get("clip_guidance", np.nan)
    gs_sort = gs if np.isfinite(gs) else 99.0
    tau_sort = tau if np.isfinite(tau) else 99.0
    clip_sort = clip if np.isfinite(clip) else 99.0
    return (
        gs_sort,
        tau_sort,
        clip_sort,
        LABEL_RANK.get(info.get("label_key"), 999),
        info.get("tag", ""),
    )


def color_for_guidance(guidance_scale, max_guidance):
    if not np.isfinite(guidance_scale):
        guidance_scale = 0.0
    if not np.isfinite(max_guidance) or max_guidance <= 0:
        max_guidance = 1.0
    frac = float(np.clip(guidance_scale / max_guidance, 0.0, 1.0))
    # Low guidance = lighter red; high guidance = darker/stronger red.
    return plt.cm.Reds(0.35 + 0.55 * frac)


def style_for_config(info):
    tau = info.get("tau_guidance", np.nan)
    clip = info.get("clip_guidance", np.nan)
    gs = info.get("guidance_scale", np.nan)
    if np.isfinite(clip) and clip <= 0:
        linestyle = ":"
    elif np.isfinite(tau) and tau <= 5:
        linestyle = "--"
    elif np.isfinite(tau) and tau >= 40:
        linestyle = "-."
    else:
        linestyle = "-"

    if not np.isfinite(gs) or gs <= 0:
        marker = "o"
    elif gs <= 0.5:
        marker = "s"
    elif gs <= 1.0:
        marker = "^"
    elif gs <= 2.0:
        marker = "D"
    elif gs <= 4.0:
        marker = "P"
    else:
        marker = "X"
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


def load_json_runs(diagnostics_root, method_dir):
    root = os.path.join(diagnostics_root, "raw", method_dir)
    runs = []
    skipped = []
    metric_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename == "metrics.json":
                metric_paths.append(os.path.join(dirpath, filename))

    for path in sorted(metric_paths):
        try:
            with open(windows_long_path(path), "r", encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as exc:
            skipped.append((path, str(exc)))
            continue
        config = payload.get("config", {})
        payload["_path"] = path
        payload["_identity"] = run_identity_from_config(config, fallback_tag=os.path.basename(os.path.dirname(os.path.dirname(path))))
        runs.append(payload)
    return runs, skipped


def group_runs(runs):
    grouped = defaultdict(list)
    info_by_tag = {}
    for run in runs:
        info = run["_identity"]
        tag = info["tag"]
        grouped[tag].append(run)
        info_by_tag[tag] = info
    return grouped, info_by_tag


def select_representative_tags(grouped, info_by_tag, preferred_labels, max_curves=6):
    ordered_tags = sorted(grouped, key=lambda tag: sort_key(info_by_tag[tag]))
    selected = []

    for label in preferred_labels:
        matches = [tag for tag in ordered_tags if info_by_tag[tag].get("label_key") == label]
        if matches:
            selected.append(matches[0])

    used_scales = {
        round(float(info_by_tag[tag].get("guidance_scale", np.nan)), 8)
        for tag in selected
        if np.isfinite(info_by_tag[tag].get("guidance_scale", np.nan))
    }
    for tag in ordered_tags:
        if tag in selected:
            continue
        gs = info_by_tag[tag].get("guidance_scale", np.nan)
        scale_key = round(float(gs), 8) if np.isfinite(gs) else None
        if scale_key not in used_scales:
            selected.append(tag)
            if scale_key is not None:
                used_scales.add(scale_key)
        if len(selected) >= max_curves:
            break

    for tag in ordered_tags:
        if len(selected) >= max_curves:
            break
        if tag not in selected:
            selected.append(tag)

    return sorted(selected, key=lambda tag: sort_key(info_by_tag[tag]))


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
):
    ordered_tags = sorted(grouped, key=lambda tag: sort_key(info_by_tag[tag]))
    if not ordered_tags:
        raise SystemExit("No readable DGBO diagnostics found.")

    max_guidance = max(
        [info_by_tag[tag].get("guidance_scale", np.nan) for tag in ordered_tags if np.isfinite(info_by_tag[tag].get("guidance_scale", np.nan))]
        or [1.0]
    )

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.2), squeeze=False)
    axes = axes[0]
    legend_handles = []
    legend_labels = []

    for ax, (metric_key, title) in zip(axes, METRICS):
        axis_values = []
        missing = []
        for tag in ordered_tags:
            info = info_by_tag[tag]
            x, mean, spread, counts = aggregate_runs(grouped[tag], metric_key, bin_size=bin_size)
            if x.size == 0 or np.all(~np.isfinite(mean)):
                if metric_key != "best_so_far_objective":
                    missing.append(short_label(info))
                continue

            color = color_for_guidance(info.get("guidance_scale", np.nan), max_guidance)
            linestyle, marker = style_for_config(info)
            label = short_label(info)
            markevery = max(1, len(x) // 12)
            line, = ax.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.25,
                marker=marker,
                markersize=4.3,
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
                        fontsize=8,
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
                fontsize=8,
                color="0.35",
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.85", linewidth=0.5),
            )
        ax.set_title(title)
        ax.set_xlabel("BO iteration")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("metric value")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(6, len(legend_labels)),
            frameon=True,
            fontsize=8.4,
            bbox_to_anchor=(0.5, -0.02),
        )

    title = "DGBO guidance sweep diagnostics"
    if title_note:
        title += f" ({title_note})"
    subtitle = "darker red = stronger guidance"
    if bin_size > 1:
        subtitle += f"; plotted in {bin_size}-iteration bins"
    if tight_diagnostic_y:
        subtitle += "; tighter diagnostic y-scale"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.90))
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), dpi=190)
    plt.close(fig)


def default_save_path(runs_root, run_group, preset, include_labels):
    filename = "dgbo_guidance_sweep_all.png"
    if preset == "representative":
        filename = "dgbo_guidance_sweep_representative.png"
    if include_labels.strip():
        filename = "dgbo_guidance_sweep_selected.png"
    return os.path.join(runs_root, run_group, "plots", filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_dgbo_sweeps")
    ap.add_argument("--method_dir", type=str, default="dgbo_latent_distillation")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs")
    ap.add_argument("--run_group", type=str, default="dgbo_latent_diffusion_sweep")
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--preset", type=str, default="all", choices=["all", "representative"])
    ap.add_argument(
        "--include_labels",
        type=str,
        default="",
        help="Comma-separated label keys, e.g. unguided,soft_clipped,base_mid_clip.",
    )
    ap.add_argument("--bin_size", type=int, default=10)
    ap.add_argument("--no_robust_y", action="store_true")
    ap.add_argument(
        "--tight_diagnostic_y",
        action="store_true",
        help="Use tighter percentile y-limits for nearest-distance and adjusted-top-k panels.",
    )
    ap.add_argument("--diagnostic_low_q", type=float, default=0.10)
    ap.add_argument("--diagnostic_high_q", type=float, default=0.90)
    args = ap.parse_args()

    runs, skipped = load_json_runs(args.diagnostics_root, args.method_dir)
    grouped, info_by_tag = group_runs(runs)

    selected_labels = []
    title_note = ""
    if args.include_labels.strip():
        selected_labels = [part.strip() for part in args.include_labels.split(",") if part.strip()]
        title_note = "selected"
    elif args.preset == "representative":
        selected_labels = REPRESENTATIVE_LABELS
        title_note = "representative subset"

    missing_selected = []
    if args.preset == "representative" and not args.include_labels.strip():
        present = {info.get("label_key") for info in info_by_tag.values()}
        missing_selected = [label for label in selected_labels if label not in present]
        selected_tags = set(select_representative_tags(grouped, info_by_tag, selected_labels, max_curves=6))
        grouped = {tag: runs_for_tag for tag, runs_for_tag in grouped.items() if tag in selected_tags}
        info_by_tag = {tag: info for tag, info in info_by_tag.items() if tag in grouped}
    elif selected_labels:
        selected = set(selected_labels)
        present = {info.get("label_key") for info in info_by_tag.values()}
        missing_selected = [label for label in selected_labels if label not in present]
        grouped = {
            tag: runs_for_tag
            for tag, runs_for_tag in grouped.items()
            if info_by_tag[tag].get("label_key") in selected
        }
        info_by_tag = {tag: info for tag, info in info_by_tag.items() if tag in grouped}

    if args.save_path is None:
        args.save_path = default_save_path(args.runs_root, args.run_group, args.preset, args.include_labels)

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
    )

    n_runs = sum(len(runs_for_tag) for runs_for_tag in grouped.values())
    print(f"Loaded {len(grouped)} configs / {n_runs} run traces.")
    if missing_selected:
        print("Representative labels not found in readable diagnostics:", ", ".join(missing_selected))
    if skipped:
        print(f"Skipped {len(skipped)} unreadable diagnostics files.")
    print("Saved:", os.path.abspath(args.save_path))


if __name__ == "__main__":
    main()
