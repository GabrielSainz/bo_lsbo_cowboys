# Toy COWBOYS & LSBO Example

Overview
--------
This repository contains a compact toy example that demonstrates two related experiments:

- COWBOYS — a simulator/experiment framework for generating and evaluating toy trajectories (the "cowboys" generative process).
- LSBO — Latent-Space Bayesian Optimization: performing Bayesian optimization inside a learned latent space (via a VAE / latent diffusion), then decoding to input space and evaluating with the COWBOYS objective.

Goals
-----
- Provide a minimal end-to-end pipeline: data generation -> latent representation learning -> optimization in latent space -> decode & evaluate.
- Make experiments reproducible and easy to inspect (saved models, traces, and generated datapoints are included in `plots/`).

Project layout (major files)
----------------------------
- [00_make_gift.py](00_make_gift.py) : small helper / demo script used in data pipeline.
- [01_data_generator_turtle_path.py](01_data_generator_turtle_path.py) : toy trajectory generator used to create training and evaluation datasets.
- [02_train_vae.py](02_train_vae.py) : script to train a VAE on generated toy trajectories.
- [02_check_vae_recon.py](02_check_vae_recon.py) : utilities to inspect reconstructions from the trained VAE.
- [03_lsbo.py](03_lsbo.py) : main script implementing Latent-Space Bayesian Optimization (LSBO).
- [04_cowboys.py](04_cowboys.py) : script implementing the COWBOYS evaluation / simulator and objective function.
- [06_train_latent_diffusion.py](06_train_latent_diffusion.py) : optional script to train a latent diffusion model (used in some experiments).
- `toy_common.py` : shared helpers and utilities used across scripts.
- `toy_data_gen.ipynb` : notebook showing interactive data generation and visualization.

Data & outputs
--------------
- `toy_circle_data/` : generated datasets and caches (e.g., `data/dataset.npz`, `latent_cache/latents.npz`).
- `models/` : trained model checkpoints (e.g., `vae.pt`, `latent_diffusion.pt`).
- `plots/` : figures, optimization traces (`bo_trace.npz`), per-experiment folders with `trace.npz` and `steps/` subfolders, and `generated_datapoints/` with metadata JSON files.

Quick start
-----------
1. Generate toy data (if you want to recreate data):

```powershell
python 01_data_generator_turtle_path.py
```

2. Train the VAE (creates `models/vae.pt`):

```powershell
python 02_train_vae.py
```

3. (Optional) Inspect reconstructions:

```powershell
python 02_check_vae_recon.py
```

4. Run Latent-Space Bayesian Optimization (LSBO):

```powershell
python 03_lsbo.py
```

5. Run / evaluate COWBOYS experiment or simulator (uses decoded candidates):

```powershell
python 04_cowboys.py
```

Configuration
-------------
- A simple `config.json` (in `toy_circle_data/`) contains experiment-level settings. Edit it to change dataset paths, model hyperparameters, or optimization settings.
- Many scripts accept command-line flags; check the top of each script for configurable parameters.

Notes & interpretation
----------------------
- The core idea is to learn a compact latent representation of toy trajectories (VAE or latent diffusion), then run Bayesian optimization in that latent space to find latent codes which decode to high-scoring trajectories under the COWBOYS objective.
- Results and diagnostics are saved under `plots/` — optimization traces, intermediate decoded samples, and metadata for generated points.
- This is a research/teaching toy: code focuses on clarity and traceability rather than production robustness.

---
Last updated: January 2026
