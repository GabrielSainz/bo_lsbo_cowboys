# python 00_make_gift.py --in_dir toy_circle_data/02_cowboys_runs/plots_cowboys_pcn_cowboys_tau1/steps --out_gif toy_circle_data/02_cowboys_runs/plots_cowboys_pcn_cowboys_tau1/cowboys.gif --fps 1

import os, re, argparse
import imageio.v2 as imageio

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="toy_circle_data/plots_lsbo/bo_steps")
    p.add_argument("--out_gif", type=str, default="toy_circle_data/plots_lsbo/lsbo.gif")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    files = sorted([f for f in os.listdir(args.in_dir) if f.lower().endswith(".png")], key=natural_key)
    if not files:
        raise RuntimeError(f"No PNG files found in {args.in_dir}")

    frames = [imageio.imread(os.path.join(args.in_dir, f)) for f in files]
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=0)
    print("Saved:", os.path.abspath(args.out_gif))

if __name__ == "__main__":
    main()
