"""Latent-space overlays for DGBO guidance-gradient variants."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_main_latent_overlays_seed import (
    encoded_data_cloud,
    ensure_dir,
    objective_norm,
    true_oracle_grid,
    windows_long_path,
)


VARIANTS = [
    {
        "slug": "direct",
        "label": "Direct DGBO",
        "guidance": "Backpropagation",
        "settings": {
            "g0p5_soft": "soft_clipped_tau10.0_gs0.5_clip5.0_cand200",
            "g1_base": "base_mid_clip_tau20.0_gs1.0_clip30.0_cand200",
            "g2_strong": "stro_default_tau20.0_gs2.0_clip30.0_cand200",
            "g4_high_tau": "ve_str_hi_tau_tau40.0_gs4.0_clip50.0_cand200",
        },
    },
    {
        "slug": "distill",
        "label": "Distillation",
        "guidance": "Learned critic",
        "settings": {
            "g0p5_soft": "distill_soft_clipped_tau10.0_gs0.5_clip5.0_cand200",
            "g1_base": "distill_base_mid_clip_tau20.0_gs1.0_clip30.0_cand200",
            "g2_strong": "distill_strong_default_tau20.0_gs2.0_clip30.0_cand200",
            "g4_high_tau": "distill_very_strong_high_tau_tau40.0_gs4.0_clip50.0",
        },
    },
    {
        "slug": "reinforce",
        "label": "REINFORCE",
        "guidance": "Score-function",
        "settings": {
            "g0p5_soft": "reinf_tau10.0_gs0.5_clip5.0",
            "g1_base": "reinf_tau20.0_gs1.0_clip30.0",
            "g2_strong": "reinf_tau20.0_gs2.0_clip30.0",
            "g4_high_tau": "reinf_very_strong_high_tau_tau40.0_gs4.0_clip50.0",
        },
    },
]


SETTING_PRESETS = {
    "soft_base": [
        {
            "slug": "g0p5_soft",
            "label": "g0.5 soft",
            "config": "soft_clipped",
            "params": "lambda=0.5, tau=10, clip=5",
        },
        {
            "slug": "g1_base",
            "label": "g1 base",
            "config": "base_mid_clip",
            "params": "lambda=1, tau=20, clip=30",
        },
    ],
    "strong": [
        {
            "slug": "g2_strong",
            "label": "g2 strong",
            "config": "strong_default",
            "params": "lambda=2, tau=20, clip=30",
        },
        {
            "slug": "g4_high_tau",
            "label": "g4 high-tau",
            "config": "very_strong_high_tau",
            "params": "lambda=4, tau=40, clip=50",
        },
    ],
}

def load_trace(runs_root, run_dir, seed):
    path = os.path.join(runs_root, run_dir, f"seed_{seed}", "trace_dgbo.npz")
    if not os.path.exists(windows_long_path(path)):
        raise FileNotFoundError(path)
    trace = np.load(windows_long_path(path), allow_pickle=True)
    return {
        "path": path,
        "z_obs": np.asarray(trace["Z_obs"], dtype=np.float64),
        "y_obs": np.asarray(trace["y_obs"], dtype=np.float64),
        "best_so_far": np.asarray(trace["best_so_far"], dtype=np.float64),
    }


def bo_points(trace, n_init):
    z = trace["z_obs"]
    y = trace["y_obs"]
    n = min(z.shape[0], y.shape[0])
    start = min(int(n_init), n)
    return z[start:n], y[start:n]


def best_bo_point(trace, n_init):
    z, y = bo_points(trace, n_init)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.full(2, np.nan)
    idxs = np.flatnonzero(finite)
    best_idx = int(idxs[np.argmax(y[finite])])
    return z[best_idx]


def build_items(runs_root, seed, settings):
    items = []
    setting_lookup = {setting["slug"]: setting for setting in settings}
    for setting in settings:
        for variant in VARIANTS:
            item = {
                "slug": f"{variant['slug']}_{setting['slug']}",
                "variant_slug": variant["slug"],
                "variant_label": variant["label"],
                "guidance": variant["guidance"],
                "setting_slug": setting["slug"],
                "setting_label": setting["label"],
                "config": setting["config"],
                "params": setting["params"],
            }
            item["trace"] = load_trace(runs_root, variant["settings"][setting["slug"]], seed)
            items.append(item)
    return items, setting_lookup


def draw_overlay(ax, item, z_data, grid, norm, z_lim, n_init):
    trace = item["trace"]
    z1, z2, grid_y = grid
    z_bo, y_bo = bo_points(trace, n_init)
    best_z = best_bo_point(trace, n_init)

    mesh = ax.contourf(
        z1,
        z2,
        grid_y,
        levels=42,
        cmap="viridis",
        norm=norm,
        alpha=0.72,
        antialiased=True,
    )
    ax.scatter(
        z_data[:, 0],
        z_data[:, 1],
        s=6,
        c="#1f77b4",
        alpha=0.11,
        linewidths=0.0,
        label="data (encoded)",
        rasterized=True,
    )
    ax.scatter(
        z_bo[:, 0],
        z_bo[:, 1],
        c=y_bo,
        cmap="viridis",
        norm=norm,
        s=38,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.94,
        label="BO points",
        zorder=4,
    )
    if np.all(np.isfinite(best_z)):
        ax.scatter(
            [best_z[0]],
            [best_z[1]],
            marker="*",
            s=250,
            c="red",
            edgecolors="black",
            linewidths=1.0,
            label="best BO z",
            zorder=6,
        )

    ax.text(
        0.03,
        0.96,
        f"{item['variant_label']}\n{item['setting_label']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="0.80", alpha=0.84, boxstyle="round,pad=0.22"),
    )
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z1", fontsize=13)
    ax.set_ylabel("z2", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(False)
    return mesh


def save_single(item, z_data, grid, norm, z_lim, n_init, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.9))
    mesh = draw_overlay(ax, item, z_data, grid, norm, z_lim, n_init)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("oracle f(decode(z))", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=11, frameon=True)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_combined(items, settings, z_data, grid, norm, z_lim, n_init, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.8), squeeze=False)
    mesh = None
    by_key = {(item["setting_slug"], item["variant_slug"]): item for item in items}
    for row, setting in enumerate(settings):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            item = by_key[(setting["slug"], variant["slug"])]
            mesh = draw_overlay(ax, item, z_data, grid, norm, z_lim, n_init)
            if row == 0:
                ax.set_title(variant["label"], fontsize=15, pad=8)

    if mesh is not None:
        cax = fig.add_axes([0.91, 0.18, 0.022, 0.70])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("oracle f(decode(z))", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12, frameon=True)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.94, wspace=0.20, hspace=0.22)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="results/toy_runs/dgbo_variants")
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_init", type=int, default=15)
    ap.add_argument("--z_lim", type=float, default=5.0)
    ap.add_argument("--background_grid_res", type=int, default=140)
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)
    ap.add_argument("--grid_oracle_path", type=str, default="00_grid_oracle.py")
    ap.add_argument(
        "--setting_preset",
        type=str,
        default="soft_base",
        choices=sorted(SETTING_PRESETS),
        help="Which pair of shared guidance settings to plot.",
    )
    args = ap.parse_args()

    z_tag = f"z{args.z_lim:g}".replace(".", "p")
    plot_dir = os.path.join(
        args.runs_root,
        "plots",
        f"latent_variant_overlays_{args.setting_preset}_seed_{args.seed}_{z_tag}",
    )
    ensure_dir(plot_dir)

    settings = SETTING_PRESETS[args.setting_preset]
    items, _ = build_items(args.runs_root, args.seed, settings)
    for item in items:
        z_bo, _ = bo_points(item["trace"], args.n_init)
        print(
            f"{item['variant_label']} / {item['setting_label']}: "
            f"{z_bo.shape[0]} BO points from {item['trace']['path']}"
        )

    z_data = encoded_data_cloud(
        args.outdir,
        n_points=args.n_data_scatter,
        seed=args.data_scatter_seed,
        grid_oracle_path=args.grid_oracle_path,
    )
    grid_cache = os.path.join(
        plot_dir,
        f"true_oracle_grid_z{args.z_lim:g}_res{args.background_grid_res}.npz",
    )
    grid = true_oracle_grid(
        args.outdir,
        z_box=args.z_lim,
        grid_res=args.background_grid_res,
        grid_oracle_path=args.grid_oracle_path,
        cache_path=grid_cache,
    )
    norm = objective_norm(grid[2], [bo_points(item["trace"], args.n_init)[1] for item in items])

    for item in items:
        stem = f"latent_overlay_{item['slug']}_seed_{args.seed}"
        save_single(item, z_data, grid, norm, args.z_lim, args.n_init, os.path.join(plot_dir, stem + ".pdf"))
        save_single(item, z_data, grid, norm, args.z_lim, args.n_init, os.path.join(plot_dir, stem + ".png"))

    save_combined(
        items,
        settings,
        z_data,
        grid,
        norm,
        args.z_lim,
        args.n_init,
        os.path.join(plot_dir, f"latent_overlay_dgbo_variants_seed_{args.seed}.pdf"),
    )
    save_combined(
        items,
        settings,
        z_data,
        grid,
        norm,
        args.z_lim,
        args.n_init,
        os.path.join(plot_dir, f"latent_overlay_dgbo_variants_seed_{args.seed}.png"),
    )
    print("Saved overlays to:", os.path.abspath(plot_dir))


if __name__ == "__main__":
    main()
