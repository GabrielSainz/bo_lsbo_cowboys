"""Latent-space overlays for selected NFlow proposal variants."""

import argparse
import json
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
        "slug": "global_lam0",
        "label": "global lam0",
        "regime": "Global flow only",
        "run_id": "global_flow_only_pi_bt1.0_tau0.10_lam0.00_sig0.20_m200_mh200",
    },
    {
        "slug": "soft_g_lam0p2",
        "label": "soft g lam0.2",
        "regime": "Mostly global mixture",
        "run_id": "soft_global_pi_bt0.5_tau0.20_lam0.20_sig0.20_m200_mh200",
    },
    {
        "slug": "bal_base_lam0p45",
        "label": "bal base lam0.45",
        "regime": "Balanced mixture",
        "run_id": "balanced_base_pi_bt1.0_tau0.10_lam0.45_sig0.20_m200_mh200",
    },
    {
        "slug": "l_tight_lam0p7",
        "label": "l-tight lam0.7",
        "regime": "Mostly local, tight",
        "run_id": "mostly_local_tight_pi_bt1.0_tau0.10_lam0.70_sig0.12_m200_mh200",
    },
    {
        "slug": "local_tight_lam1",
        "label": "local tight lam1",
        "regime": "Local only, tight",
        "run_id": "local_only_tight_pi_bt1.0_tau0.10_lam1.00_sig0.12_m200_mh200",
    },
]


def load_json(path):
    with open(windows_long_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def find_metrics(diagnostics_root, run_id, seed):
    for dirpath, _, filenames in os.walk(diagnostics_root):
        if "metrics.json" not in filenames:
            continue
        path = os.path.join(dirpath, "metrics.json")
        payload = load_json(path)
        config = payload.get("config", {})
        if int(config.get("seed", -1)) != int(seed):
            continue
        if config.get("diagnostics_run_id") == run_id:
            return payload, path
    raise FileNotFoundError(f"No metrics.json found for run_id={run_id!r}, seed={seed}")


def selected_points(payload):
    z = []
    y = []
    for row in payload.get("iterations", []):
        selected_z = row.get("selected_latent")
        selected_y = row.get("selected_objective")
        if selected_z is None or selected_y is None:
            continue
        z.append(selected_z)
        y.append(selected_y)
    return np.asarray(z, dtype=np.float64).reshape(-1, 2), np.asarray(y, dtype=np.float64)


def best_bo_point(payload):
    z, y = selected_points(payload)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.full(2, np.nan)
    idxs = np.flatnonzero(finite)
    best_idx = int(idxs[np.argmax(y[finite])])
    return z[best_idx]


def draw_overlay(ax, item, z_data, grid, norm, z_lim):
    payload = item["payload"]
    z1, z2, grid_y = grid
    z_bo, y_bo = selected_points(payload)
    best_z = best_bo_point(payload)

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
        f"{item['label']}\n{item['regime']}",
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


def save_single(item, z_data, grid, norm, z_lim, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.9))
    mesh = draw_overlay(ax, item, z_data, grid, norm, z_lim)
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


def save_combined(items, z_data, grid, norm, z_lim, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.8), squeeze=False)
    axes_flat = axes.ravel()
    mesh = None
    for ax, item in zip(axes_flat, items):
        mesh = draw_overlay(ax, item, z_data, grid, norm, z_lim)
    axes_flat[-1].axis("off")
    if mesh is not None:
        cax = fig.add_axes([0.91, 0.18, 0.022, 0.70])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("oracle f(decode(z))", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12, frameon=True)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.98, wspace=0.20, hspace=0.20)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_sweeps/raw/cowboys_flow_latent")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs/cowboys_flow_latent_proposal_sweep")
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--z_lim", type=float, default=4.0)
    ap.add_argument("--background_grid_res", type=int, default=140)
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)
    ap.add_argument("--grid_oracle_path", type=str, default="00_grid_oracle.py")
    args = ap.parse_args()

    z_tag = f"z{args.z_lim:g}".replace(".", "p")
    plot_dir = os.path.join(args.runs_root, "plots", f"latent_nflow_overlays_seed_{args.seed}_{z_tag}")
    ensure_dir(plot_dir)

    items = []
    for spec in VARIANTS:
        payload, path = find_metrics(args.diagnostics_root, spec["run_id"], args.seed)
        item = dict(spec)
        item["payload"] = payload
        item["path"] = path
        z_bo, _ = selected_points(payload)
        print(f"{item['label']}: {z_bo.shape[0]} BO points from {path}")
        items.append(item)

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
    norm = objective_norm(grid[2], [selected_points(item["payload"])[1] for item in items])

    for item in items:
        stem = f"latent_overlay_{item['slug']}_seed_{args.seed}"
        save_single(item, z_data, grid, norm, args.z_lim, os.path.join(plot_dir, stem + ".pdf"))
        save_single(item, z_data, grid, norm, args.z_lim, os.path.join(plot_dir, stem + ".png"))

    save_combined(
        items,
        z_data,
        grid,
        norm,
        args.z_lim,
        os.path.join(plot_dir, f"latent_overlay_nflow_selected_seed_{args.seed}.pdf"),
    )
    save_combined(
        items,
        z_data,
        grid,
        norm,
        args.z_lim,
        os.path.join(plot_dir, f"latent_overlay_nflow_selected_seed_{args.seed}.png"),
    )
    print("Saved overlays to:", os.path.abspath(plot_dir))


if __name__ == "__main__":
    main()
