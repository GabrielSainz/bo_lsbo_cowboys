# python 06_train_latent_diffusion.py --outdir toy_circle_data --cache_latents --steps 20000 --T 200

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Utils
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(outdir: str):
    cfg_json = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_json):
        with open(cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg
    return {}


# =========================
# VAE (must match your trained checkpoint)
# =========================
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class ResMLP(nn.Module):
    def __init__(self, d_in, d_hidden, n_layers=4, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([MLPBlock(d_hidden, d_hidden, dropout) for _ in range(n_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_hidden)

    def forward(self, x):
        h = self.in_proj(x)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(ln(h))
        return self.out_norm(h)

class VAE(nn.Module):
    def __init__(self, L: int, latent_dim: int, hidden: int = 512,
                 enc_layers: int = 6, dec_layers: int = 6, dropout: float = 0.05):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim

        self.encoder = ResMLP(L, hidden, n_layers=enc_layers, dropout=dropout)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        dec_blocks = []
        for _ in range(dec_layers):
            dec_blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden),
            ))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            *dec_blocks,
            nn.Linear(hidden, L),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(z)

def load_vae(vae_path: str, device):
    ckpt = torch.load(vae_path, map_location=device)
    L = ckpt["L"]
    latent_dim = ckpt["latent_dim"]
    hidden = ckpt.get("hidden", 512)
    enc_layers = ckpt.get("enc_layers", 6)
    dec_layers = ckpt.get("dec_layers", 6)
    dropout = ckpt.get("dropout", 0.05)

    vae = VAE(L=L, latent_dim=latent_dim, hidden=hidden,
              enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout).to(device)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.eval()
    return vae, L, latent_dim


@torch.no_grad()
def encode_mu_batch(vae, X_radians: np.ndarray, device, batch_size=1024):
    """
    X_radians: (N,L) in radians.
    Return Z_mu: (N,latent_dim)
    """
    X_scaled = (X_radians / np.pi).astype(np.float32)  # VAE expects [-1,1]
    Xt = torch.from_numpy(X_scaled).to(device)
    Z = []
    for i in range(0, Xt.shape[0], batch_size):
        mu, _ = vae.encode(Xt[i:i+batch_size])
        Z.append(mu.cpu().numpy())
    return np.vstack(Z)


# =========================
# Diffusion model in latent space (DDPM)
# =========================
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B,) integer
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class EpsMLP(nn.Module):
    def __init__(self, z_dim: int, time_dim: int = 32, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.temb = SinusoidalTimeEmb(time_dim)
        layers = []
        d_in = z_dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        layers += [nn.Linear(hidden, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        te = self.temb(t)
        x = torch.cat([zt, te], dim=1)
        return self.net(x)

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)

@torch.no_grad()
def ddpm_sample(model, betas, n_samples, device):
    """
    Simple DDPM ancestral sampler (unconditional).
    Returns z0 samples in normalized space, and snapshots of predicted z0 at a few t.
    """
    T = len(betas)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = 1.0 - betas_t
    abar = torch.cumprod(alphas, dim=0)

    z = torch.randn(n_samples, 2, device=device)
    keep_ts = sorted({T-1, int(0.75*(T-1)), int(0.5*(T-1)), int(0.25*(T-1)), 0})
    snaps = {}

    for t in reversed(range(T)):
        tt = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps_hat = model(z, tt)

        ab = abar[t]
        a = alphas[t]

        # predicted clean z0
        z0_hat = (z - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)

        # store snapshot of z0_hat (not z_t)
        if t in keep_ts:
            snaps[t] = z0_hat.detach().cpu().numpy()

        # DDPM update
        if t > 0:
            noise = torch.randn_like(z)
            sigma = torch.sqrt(betas_t[t])
        else:
            noise = torch.zeros_like(z)
            sigma = 0.0

        z = (1.0 / torch.sqrt(a)) * (z - (betas_t[t] / torch.sqrt(1 - ab)) * eps_hat) + sigma * noise

    # final z is z_0-ish but in this implementation we return z0_hat from last step
    return z.detach().cpu().numpy(), snaps


# =========================
# Plots
# =========================
def plot_latent_types(Z, types, save_path):
    plt.figure(figsize=(6.5, 5.5))
    uniq = np.unique(types)
    for u in uniq:
        m = (types == u)
        plt.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55, label=f"type {int(u)}")
    plt.title("VAE latent means colored by type")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()

def plot_latent_oracle(Z, y, save_path, title="VAE latent means colored by oracle"):
    plt.figure(figsize=(6.5, 5.5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=12, alpha=0.8)
    plt.colorbar(sc, label="oracle y")
    plt.title(title)
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()

def plot_forward_noising(Zn, T, betas, save_path, n_show=4500):
    """
    Zn: normalized Z0.
    Show how q(z_t|z0) destroys structure.
    """
    N = Zn.shape[0]
    idx = np.random.choice(N, size=min(n_show, N), replace=False)
    Z0 = Zn[idx]

    alphas = 1.0 - betas
    abar = np.cumprod(alphas)

    ts = [0, int(0.1*(T-1)), int(0.25*(T-1)), int(0.5*(T-1)), int(0.75*(T-1)), T-1]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    axes = axes.flatten()
    for ax, t in zip(axes, ts):
        eps = np.random.randn(*Z0.shape).astype(np.float32)
        zt = np.sqrt(abar[t]).astype(np.float32) * Z0 + np.sqrt(1.0 - abar[t]).astype(np.float32) * eps

        ax.scatter(zt[:,0], zt[:,1], s=8, alpha=0.35, linewidth=0.0)
        ax.set_title(f"Forward noise: t={t}")
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.axhline(0.0, linewidth=0.5)
        ax.axvline(0.0, linewidth=0.5)

    plt.suptitle("Forward diffusion in latent space (normalized z)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close(fig)

def plot_reverse_samples(Zn_data, Zn_samp, save_path):
    """
    Compare generated samples to data, in normalized space.
    """
    plt.figure(figsize=(6.8, 6.0))
    plt.scatter(Zn_data[:,0], Zn_data[:,1], s=10, alpha=0.18, linewidth=0.0, label="data z0 (encoded)")
    plt.scatter(Zn_samp[:,0], Zn_samp[:,1], s=12, alpha=0.35, linewidth=0.0, label="diffusion samples")
    plt.title("Reverse diffusion samples vs encoded latent data (normalized)")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()

def plot_reverse_snapshots(Zn_data, snaps, save_path):
    keys = sorted(snaps.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4*len(keys), 3.4))
    if len(keys) == 1:
        axes = [axes]
    for ax, t in zip(axes, keys):
        ax.scatter(Zn_data[:,0], Zn_data[:,1], s=10, alpha=0.15, linewidth=0.0)
        z = snaps[t]
        ax.scatter(z[:,0], z[:,1], s=10, alpha=0.35, linewidth=0.0)
        ax.set_title(f"pred z0 @ t={t}")
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
    plt.suptitle("Reverse diffusion snapshots (predicted z0 at different t)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close(fig)

def plot_reverse_trajectories(model, betas, device, n_traj, save_path):
    """
    Track a few points through reverse process (z_t path).
    """
    T = len(betas)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = 1.0 - betas_t
    abar = torch.cumprod(alphas, dim=0)

    z = torch.randn(n_traj, 2, device=device)
    traj = [z.detach().cpu().numpy()]

    for t in reversed(range(T)):
        tt = torch.full((n_traj,), t, device=device, dtype=torch.long)
        eps_hat = model(z, tt)

        ab = abar[t]
        a = alphas[t]

        if t > 0:
            noise = torch.randn_like(z)
            sigma = torch.sqrt(betas_t[t])
        else:
            noise = torch.zeros_like(z)
            sigma = 0.0

        z = (1.0 / torch.sqrt(a)) * (z - (betas_t[t] / torch.sqrt(1 - ab)) * eps_hat) + sigma * noise
        if t in {T-1, int(0.8*(T-1)), int(0.6*(T-1)), int(0.4*(T-1)), int(0.2*(T-1)), 0}:
            traj.append(z.detach().cpu().numpy())

    traj = np.stack(traj, axis=0)  # (K, n_traj, 2)

    plt.figure(figsize=(6.8, 6.0))
    for i in range(n_traj):
        plt.plot(traj[:,i,0], traj[:,i,1], marker="o", linewidth=1.6, markersize=4)
        plt.scatter(traj[0,i,0], traj[0,i,1], s=80, marker="s", edgecolor="black", label="start" if i==0 else None)
        plt.scatter(traj[-1,i,0], traj[-1,i,1], s=120, marker="*", edgecolor="black", label="end" if i==0 else None)
    plt.title("A few reverse diffusion trajectories (normalized z)")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="toy_circle_data")
    ap.add_argument("--vae_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    # encoding
    ap.add_argument("--n_encode", type=int, default=25000, help="How many x points to encode to latent for diffusion.")
    ap.add_argument("--encode_batch", type=int, default=1024)
    ap.add_argument("--cache_latents", action="store_true")

    # diffusion
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)

    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)

    # plotting
    ap.add_argument("--n_scatter", type=int, default=6000)
    ap.add_argument("--n_gen", type=int, default=4000)
    ap.add_argument("--n_traj", type=int, default=6)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = args.outdir
    data_npz = os.path.join(outdir, "data", "dataset.npz")
    assert os.path.exists(data_npz), f"Missing dataset: {data_npz}"
    data = np.load(data_npz, allow_pickle=True)

    X = data["X"].astype(np.float64)
    types = data["types"].astype(np.int64) if "types" in data else None
    y = data["y"].astype(np.float64) if "y" in data else None

    vae_path = args.vae_path or os.path.join(outdir, "models", "vae.pt")
    assert os.path.exists(vae_path), f"Missing VAE: {vae_path}"
    vae, L, z_dim = load_vae(vae_path, device)
    assert z_dim == 2, "This visualization script assumes latent_dim=2."

    # -------------------------
    # Encode to latent means
    # -------------------------
    N = X.shape[0]
    n_enc = min(args.n_encode, N)
    idx = np.random.choice(N, size=n_enc, replace=False)
    X_enc = X[idx]
    types_enc = types[idx] if types is not None else None
    y_enc = y[idx] if y is not None else None

    print(f"Encoding {n_enc} points to latent means...")
    Z = encode_mu_batch(vae, X_enc, device, batch_size=args.encode_batch).astype(np.float32)

    # normalize Z for diffusion stability
    z_mean = Z.mean(axis=0, keepdims=True)
    z_std = Z.std(axis=0, keepdims=True)
    z_std = np.clip(z_std, 1e-6, None)
    Zn = (Z - z_mean) / z_std

    # optional cache
    if args.cache_latents:
        lat_dir = os.path.join(outdir, "latent_cache")
        ensure_dir(lat_dir)
        np.savez_compressed(
            os.path.join(lat_dir, "latents.npz"),
            Z=Z, Zn=Zn, z_mean=z_mean, z_std=z_std,
            idx=idx, types=types_enc if types_enc is not None else np.array([-1]),
            y=y_enc if y_enc is not None else np.array([0.0]),
        )

    # -------------------------
    # Make initial plots (VAE latent)
    # -------------------------
    plot_dir = os.path.join(outdir, "plots_latent_diffusion")
    ensure_dir(plot_dir)

    # scatter subset for plots
    ns = min(args.n_scatter, Zn.shape[0])
    sidx = np.random.choice(Zn.shape[0], size=ns, replace=False)
    Zp = Z[sidx]
    Znp = Zn[sidx]
    tp = types_enc[sidx] if types_enc is not None else None
    yp = y_enc[sidx] if y_enc is not None else None

    if tp is not None:
        plot_latent_types(Zp, tp, os.path.join(plot_dir, "latent_types.png"))

    if yp is not None:
        plot_latent_oracle(Zp, yp, os.path.join(plot_dir, "latent_oracle.png"),
                           title="VAE latent means colored by oracle (from dataset)")

    # forward noising visualization in normalized space
    betas = make_beta_schedule(args.T, args.beta_start, args.beta_end).astype(np.float32)
    plot_forward_noising(Znp, args.T, betas, os.path.join(plot_dir, "forward_noising_grid.png"), n_show=ns)

    # -------------------------
    # Train diffusion model on Zn
    # -------------------------
    model = EpsMLP(z_dim=2, time_dim=32, hidden=256, depth=4).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    Zn_t = torch.from_numpy(Zn).to(device)

    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    sqrt_abar = torch.tensor(np.sqrt(abar).astype(np.float32), device=device)
    sqrt_1mabar = torch.tensor(np.sqrt(1.0 - abar).astype(np.float32), device=device)

    print(f"Training latent diffusion: T={args.T}, steps={args.steps}, batch={args.batch}")
    losses = []
    model.train()

    for step in range(1, args.steps + 1):
        ids = torch.randint(0, Zn_t.shape[0], (args.batch,), device=device)
        z0 = Zn_t[ids]
        t = torch.randint(0, args.T, (args.batch,), device=device)  # 0..T-1
        eps = torch.randn_like(z0)

        zt = sqrt_abar[t].unsqueeze(1) * z0 + sqrt_1mabar[t].unsqueeze(1) * eps
        eps_hat = model(zt, t)

        loss = torch.mean((eps_hat - eps) ** 2)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))
        if step == 1 or step % 500 == 0:
            print(f"step {step:5d} | loss {loss.item():.6f}")

    # save loss plot
    plt.figure(figsize=(7,4))
    plt.plot(losses)
    plt.title("Diffusion training loss (MSE on noise)")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "train_loss.png"), dpi=170)
    plt.close()

    # save checkpoint
    ckpt = {
        "T": args.T,
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "state_dict": model.state_dict(),
        "z_mean": z_mean.astype(np.float32),
        "z_std": z_std.astype(np.float32),
        "z_dim": 2,
        "time_dim": 32,
        "hidden": 256,
        "depth": 4,
        "seed": args.seed,
    }
    model_path = os.path.join(outdir, "models", "latent_diffusion.pt")
    ensure_dir(os.path.dirname(model_path))
    torch.save(ckpt, model_path)
    print("Saved diffusion model:", os.path.abspath(model_path))

    # -------------------------
    # Reverse sampling checks + plots
    # -------------------------
    model.eval()
    Zn_samp, snaps = ddpm_sample(model, betas, n_samples=args.n_gen, device=device)

    # compare with data (normalized)
    plot_reverse_samples(Znp, Zn_samp, os.path.join(plot_dir, "reverse_samples_overlay.png"))
    plot_reverse_snapshots(Znp, snaps, os.path.join(plot_dir, "reverse_snapshots.png"))
    plot_reverse_trajectories(model, betas, device, n_traj=args.n_traj,
                              save_path=os.path.join(plot_dir, "reverse_trajectories.png"))

    # one more creative plot: unnormalize and overlay (real z-space)
    Z_samp = Zn_samp * z_std + z_mean
    plt.figure(figsize=(6.8, 6.0))
    plt.scatter(Zp[:,0], Zp[:,1], s=10, alpha=0.18, linewidth=0.0, label="encoded z (data)")
    plt.scatter(Z_samp[:,0], Z_samp[:,1], s=12, alpha=0.35, linewidth=0.0, label="diffusion samples")
    plt.title("Encoded latent vs diffusion samples (real z-space)")
    plt.xlabel("z1"); plt.ylabel("z2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reverse_samples_real_z.png"), dpi=170)
    plt.close()

    print("Saved plots to:", os.path.abspath(plot_dir))
    print("Key plots:")
    print(" - latent_types.png, latent_oracle.png")
    print(" - forward_noising_grid.png")
    print(" - reverse_samples_overlay.png, reverse_snapshots.png, reverse_trajectories.png")


if __name__ == "__main__":
    main()
