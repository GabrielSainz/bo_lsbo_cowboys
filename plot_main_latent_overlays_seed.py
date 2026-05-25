"""Consistent latent-space overlays for one seed of the main toy benchmark."""

import argparse
import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


METHOD_SPECS = [
    ("lsbo", "LSBO", "lsbo/seed_{seed}/metrics.json"),
    ("cowboys", "COWBOYS", "cowboys/seed_{seed}/metrics.json"),
    ("dgbo", "DGBO", "dgbo_latent_diffusion/seed_{seed}/metrics.json"),
    (
        "nflow",
        "NFlow",
        "nflows/{nflow_run_dir}/seed_{seed}/metrics.json",
    ),
]


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


def load_json(path):
    with open(windows_long_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def json_array(value, width=None):
    arr = np.asarray(value or [], dtype=np.float64)
    if arr.size == 0:
        if width is None:
            return np.zeros((0,), dtype=np.float64)
        return np.zeros((0, width), dtype=np.float64)
    if width is not None:
        arr = arr.reshape(-1, width)
    return arr


def load_grid_oracle_module(script_path):
    spec = importlib.util.spec_from_file_location("grid_oracle_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded_data_cloud(outdir, n_points, seed, grid_oracle_path):
    module = load_grid_oracle_module(grid_oracle_path)
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    data = np.load(windows_long_path(data_npz), allow_pickle=True)
    x = data["X"].astype(np.float64)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = os.path.join(outdir, "models", "vae.pt")
    vae, _, latent_dim = module.load_vae(vae_path, device)
    if latent_dim != 2:
        raise ValueError(f"Latent overlay expects latent_dim=2, got {latent_dim}")
    return module.encode_dataset_to_z(vae, x, device, n_points=n_points, seed=seed)


def true_oracle_grid(outdir, z_box, grid_res, grid_oracle_path, cache_path):
    if cache_path and os.path.exists(windows_long_path(cache_path)):
        cached = np.load(windows_long_path(cache_path))
        return cached["z1"], cached["z2"], cached["objective_grid"]

    module = load_grid_oracle_module(grid_oracle_path)
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    data = np.load(windows_long_path(data_npz), allow_pickle=True)
    target = data["target"].astype(np.float64)

    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(windows_long_path(cfg_json)):
        with open(windows_long_path(cfg_json), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        step_size = float(cfg.get("step_size", 1.0))
        w_close = float(cfg.get("w_close", 0.2))
        w_smooth = float(cfg.get("w_smooth", 0.05))
    else:
        step_size, w_close, w_smooth = 1.0, 0.2, 0.05

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = os.path.join(outdir, "models", "vae.pt")
    vae, _, latent_dim = module.load_vae(vae_path, device)
    if latent_dim != 2:
        raise ValueError(f"Latent overlay expects latent_dim=2, got {latent_dim}")

    z1 = np.linspace(-float(z_box), float(z_box), int(grid_res))
    z2 = np.linspace(-float(z_box), float(z_box), int(grid_res))
    grid_x, grid_y = np.meshgrid(z1, z2)
    z_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    x_chunks = []
    decode_bs = 4096
    for start in range(0, z_grid.shape[0], decode_bs):
        x_chunks.append(module.decode_batch_z_to_x_radians(vae, z_grid[start : start + decode_bs], device))
    x_grid = np.vstack(x_chunks)
    y_grid = module.oracle_f_batch(x_grid, target, step_size, w_close=w_close, w_smooth=w_smooth)
    objective_grid = y_grid.reshape(int(grid_res), int(grid_res))

    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        np.savez_compressed(
            windows_long_path(cache_path),
            z1=z1,
            z2=z2,
            objective_grid=objective_grid,
        )
    return z1, z2, objective_grid


def load_run(diagnostics_root, seed, rel_path, nflow_run_dir):
    raw_root = os.path.join(diagnostics_root, "raw")
    path = os.path.join(raw_root, rel_path.format(seed=seed, nflow_run_dir=nflow_run_dir))
    return load_json(path), path


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
    z = json_array(z, width=2)
    y = json_array(y)
    return z, y


def best_point(payload, z_bo, y_bo):
    summary = payload.get("summary", {})
    if summary.get("best_latent") is not None:
        return np.asarray(summary["best_latent"], dtype=np.float64).reshape(2)
    if y_bo.size:
        return z_bo[int(np.nanargmax(y_bo))]
    return np.full(2, np.nan)


def background_grid(reference_payload):
    bg = reference_payload.get("objective_background") or {}
    z1 = json_array(bg.get("z1"))
    z2 = json_array(bg.get("z2"))
    y = np.asarray(bg.get("objective_grid") or [], dtype=np.float64)
    return z1, z2, y


def objective_norm(grid_y, all_objectives, q_low=0.02, q_high=0.98):
    values = []
    grid_vals = np.asarray(grid_y, dtype=np.float64)
    grid_vals = grid_vals[np.isfinite(grid_vals)]
    if grid_vals.size:
        values.append(grid_vals)
    for y in all_objectives:
        vals = np.asarray(y, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            values.append(vals)
    if not values:
        return Normalize(vmin=-10.0, vmax=0.0)
    vals = np.concatenate(values)
    return Normalize(vmin=float(np.quantile(vals, q_low)), vmax=float(np.quantile(vals, q_high)))


def draw_overlay(
    ax,
    payload,
    method_label,
    z_data,
    grid,
    norm,
    z_lim,
    show_colorbar=False,
    fig=None,
):
    z1, z2, grid_y = grid
    z_bo, y_bo = selected_points(payload)
    best_z = best_point(payload, z_bo, y_bo)

    mesh = None
    if z1.size and z2.size and grid_y.size:
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

    if z_data.size:
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

    if z_bo.size:
        ax.scatter(
            z_bo[:, 0],
            z_bo[:, 1],
            c=y_bo,
            cmap="viridis",
            norm=norm,
            s=36,
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
            s=245,
            c="red",
            edgecolors="black",
            linewidths=1.0,
            label="best BO z",
            zorder=6,
        )

    ax.text(
        0.03,
        0.96,
        method_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="0.80", alpha=0.82, boxstyle="round,pad=0.22"),
    )
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z1", fontsize=13)
    ax.set_ylabel("z2", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(False)

    if show_colorbar and fig is not None and mesh is not None:
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("oracle f(decode(z))", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    return mesh


def save_single_plot(payload, method_label, z_data, grid, norm, z_lim, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.9))
    draw_overlay(
        ax,
        payload,
        method_label,
        z_data,
        grid,
        norm,
        z_lim,
        show_colorbar=True,
        fig=fig,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=11, frameon=True)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_combined_plot(runs, z_data, grid, norm, z_lim, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.4), squeeze=False)
    axes = axes.ravel()
    mesh = None
    for ax, (_, label, payload) in zip(axes, runs):
        drawn = draw_overlay(ax, payload, label, z_data, grid, norm, z_lim)
        if drawn is not None:
            mesh = drawn
    if mesh is not None:
        cax = fig.add_axes([0.90, 0.19, 0.026, 0.68])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("oracle f(decode(z))", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.10, top=0.98, wspace=0.18, hspace=0.18)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(windows_long_path(save_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_root", type=str, default="results/toy_diagnostics_main_analysis")
    ap.add_argument("--runs_root", type=str, default="results/toy_runs/main_comparison")
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nflow_run_dir", type=str, default="balance_fcc898bf")
    ap.add_argument("--z_lim", type=float, default=4.0)
    ap.add_argument("--n_data_scatter", type=int, default=4000)
    ap.add_argument("--data_scatter_seed", type=int, default=0)
    ap.add_argument("--background_grid_res", type=int, default=140)
    ap.add_argument("--grid_oracle_path", type=str, default="00_grid_oracle.py")
    args = ap.parse_args()

    loaded = []
    for slug, label, rel_path in METHOD_SPECS:
        payload, path = load_run(args.diagnostics_root, args.seed, rel_path, args.nflow_run_dir)
        loaded.append((slug, label, payload))
        run_id = payload.get("config", {}).get("diagnostics_run_id")
        print(f"{label}: {path}" + (f" ({run_id})" if run_id else ""))

    z_data = encoded_data_cloud(
        args.outdir,
        n_points=args.n_data_scatter,
        seed=args.data_scatter_seed,
        grid_oracle_path=args.grid_oracle_path,
    )
    plot_dir = os.path.join(args.runs_root, "plots", f"latent_overlays_seed_{args.seed}")
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
    norm = objective_norm(grid[2], [selected_points(payload)[1] for _, _, payload in loaded])

    ensure_dir(plot_dir)

    for slug, label, payload in loaded:
        save_single_plot(
            payload,
            label,
            z_data,
            grid,
            norm,
            args.z_lim,
            os.path.join(plot_dir, f"latent_overlay_{slug}_seed_{args.seed}.pdf"),
        )
        save_single_plot(
            payload,
            label,
            z_data,
            grid,
            norm,
            args.z_lim,
            os.path.join(plot_dir, f"latent_overlay_{slug}_seed_{args.seed}.png"),
        )

    save_combined_plot(
        loaded,
        z_data,
        grid,
        norm,
        args.z_lim,
        os.path.join(plot_dir, f"latent_overlay_all_methods_seed_{args.seed}.pdf"),
    )
    save_combined_plot(
        loaded,
        z_data,
        grid,
        norm,
        args.z_lim,
        os.path.join(plot_dir, f"latent_overlay_all_methods_seed_{args.seed}.png"),
    )

    print("Saved overlays to:", os.path.abspath(plot_dir))


if __name__ == "__main__":
    main()
