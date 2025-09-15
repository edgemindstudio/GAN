# gan/sample.py

"""
Sampling utilities for the Conditional DCGAN.

Features
--------
- Loads generator architecture via `gan.models.build_models` and weights from a checkpoint.
- Generates synthetic samples either:
    * balanced: N samples per class, or
    * custom: total count split across a subset / explicit per-class counts.
- Saves per-class `.npy` arrays for traceability, and an optional preview grid `.png`.
- Outputs a small metadata JSON alongside samples for reproducibility.

Examples
--------
# Balanced: 1000 samples per class, save to default artifacts dir
python -m gan.sample --samples-per-class 1000

# Total 5000 samples, balanced across all 9 classes
python -m gan.sample --num-samples 5000 --balanced

# Only classes 0,1,3 with custom counts (e.g., 100,200,50)
python -m gan.sample --per-class 0:100 1:200 3:50

# Custom weights path and preview grid
python -m gan.sample --weights ./artifacts/checkpoints/generator_best.h5 --grid 9 20

Notes
-----
- Generator is assumed to output in [-1, 1] (tanh). This script rescales to [0,1].
- `config.yaml` must define IMG_SHAPE, NUM_CLASSES, LATENT_DIM (or they fall back to defaults).
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Optional
# Local imports
from gan.models import build_models

# -----------------------------
# Utilities
# -----------------------------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{now_ts()}] {msg}")

def make_dirs(root: Path):
    (root / "artifacts" / "samples").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parsing helpers
# -----------------------------
def parse_per_class(arg_list: list[str]) -> dict[int, int]:
    """
    Parse items like ["0:100", "3:50"] -> {0:100, 3:50}
    """
    out: dict[int, int] = {}
    for item in arg_list:
        try:
            k, v = item.split(":")
            k, v = int(k), int(v)
            if v <= 0:
                raise ValueError
            out[k] = v
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid --per-class entry '{item}'. Use 'label:count' with positive count."
            )
    return out

def parse_classes(arg_list: list[str]) -> list[int]:
    try:
        return [int(x) for x in arg_list]
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --classes entries; must be integers.")

# -----------------------------
# Latent sampling
# -----------------------------
def sample_latents(n: int, dim: int, truncation: float | None = None) -> np.ndarray:
    """
    Sample latent vectors. If `truncation` is provided (>0), clip to [-t, t].
    """
    z = np.random.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    if truncation is not None and truncation > 0:
        z = np.clip(z, -float(truncation), float(truncation))
    return z

# -----------------------------
# Image utilities
# -----------------------------
def to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert [0,1] float to uint8 [0,255] safely."""
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)

def save_grid(
    images01: np.ndarray,
    img_shape: tuple[int, int, int],
    rows: int,
    cols: int,
    out_path: Path,
    titles: list[str] | None = None,
):
    """
    Save a grid of images in [0,1]. Supports 1 or 3 channels.
    """
    H, W, C = img_shape
    n = min(images01.shape[0], rows * cols)
    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im = images01[i].reshape(H, W, C)
        if C == 1:
            plt.imshow(im.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(im, 0.0, 1.0))
        if titles:
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------------
# Build & load generator
# -----------------------------
def build_and_load_generator(
    latent_dim: int,
    num_classes: int,
    img_shape: tuple[int, int, int],
    weights_path: Path | None,
    lr: float = 2e-4,
    beta_1: float = 0.5,
) -> tf.keras.Model:
    """
    Reconstruct generator architecture via gan.models.build_models and load weights if provided.
    """
    models_dict = build_models(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_shape=img_shape,
        lr=lr,
        beta_1=beta_1,
    )
    G = models_dict["generator"]
    if weights_path is not None and weights_path.exists():
        G.load_weights(str(weights_path))
        log(f"Loaded generator weights: {weights_path}")
    else:
        if weights_path is not None:
            log(f"WARNING: weights not found at {weights_path}. Continuing with untrained generator.")
        else:
            log("WARNING: no weights path provided. Continuing with untrained generator.")
    return G

# -----------------------------
# Main sampling logic
# -----------------------------
def generate_for_classes(
    G: tf.keras.Model,
    latent_dim: int,
    img_shape: tuple[int, int, int],
    class_counts: dict[int, int],
    num_classes: int,
    save_dir: Path,
    save_per_class_npy: bool = True,
    truncation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate per-class images and return concatenated arrays:
        x_synth_01: (N, H, W, C) in [0,1]
        y_synth_oh: (N, num_classes) one-hot
    Also optionally saves per-class .npy dumps for traceability.
    """
    H, W, C = img_shape
    xs, ys = [], []

    for cls, count in class_counts.items():
        if cls < 0 or cls >= num_classes:
            raise ValueError(f"Class {cls} is out of range [0, {num_classes-1}]")
        if count <= 0:
            continue

        z = sample_latents(count, latent_dim, truncation=truncation)
        y = tf.keras.utils.to_categorical(np.full((count, 1), cls), num_classes).astype(np.float32)
        g = G.predict([z, y], verbose=0)              # expected in [-1,1]
        g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)      # -> [0,1]

        xs.append(g01.reshape(-1, H, W, C))
        ys.append(y)

        if save_per_class_npy:
            np.save(save_dir / f"gen_class_{cls}.npy", g01)
            np.save(save_dir / f"labels_class_{cls}.npy", np.full((count,), cls, dtype=np.int32))
            log(f"Saved class {cls} -> {count} samples to {save_dir}")

    if not xs:
        return np.empty((0, H, W, C), dtype=np.float32), np.empty((0, num_classes), dtype=np.float32)

    x_synth = np.concatenate(xs, axis=0)
    y_synth = np.concatenate(ys, axis=0)
    return x_synth, y_synth

# -----------------------------
# CLI
# -----------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Sample synthetic data from a trained Conditional DCGAN.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to generator weights (defaults to artifacts/checkpoints/generator_best.h5)")
    # Sampling modes
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="If set, generate this many samples per class (balanced).")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Total number of samples to generate. Requires --balanced or --classes.")
    parser.add_argument("--balanced", action="store_true",
                        help="When using --num-samples, split evenly across classes (or selected classes).")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Subset of class IDs to sample (e.g., --classes 0 1 3). Defaults to all classes.")
    parser.add_argument("--per-class", nargs="+", default=None,
                        help="Explicit per-class counts, e.g., --per-class 0:100 2:50")

    # Output & preview
    parser.add_argument("--outdir", type=str, default=None,
                        help="Override output directory (default: artifacts/samples)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional subfolder tag under samples/, e.g., 'exp1'")
    parser.add_argument("--grid", nargs=2, type=int, default=None,
                        help="Save a preview grid: ROWS COLS, e.g., --grid 6 9")
    parser.add_argument("--grid-path", type=str, default=None,
                        help="Override grid output path (default: samples/<tag>/preview_grid.png)")
    parser.add_argument("--no-save-per-class", action="store_true",
                        help="Do not save per-class .npy dumps (still returns arrays).")

    # Sampling behavior
    parser.add_argument("--truncation", type=float, default=None,
                        help="Clip latent z to [-t, t]. If omitted, use full N(0,1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args(argv)

    # Setup
    root = Path(__file__).resolve().parents[1]
    make_dirs(root)
    set_seeds(args.seed)
    enable_gpu_memory_growth()

    # Config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    IMG_SHAPE   = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    NUM_CLASSES = int(cfg.get("NUM_CLASSES", 9))
    LATENT_DIM  = int(cfg.get("LATENT_DIM", 100))
    LR          = float(cfg.get("LR", 2e-4))
    BETA_1      = float(cfg.get("BETA_1", 0.5))

    # Output dirs
    samples_root = Path(args.outdir) if args.outdir else (root / "artifacts" / "samples")
    if args.tag:
        samples_dir = samples_root / args.tag
    else:
        # timestamped folder to avoid accidental overwrite
        samples_dir = samples_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    weights_path = Path(args.weights) if args.weights else (root / "artifacts" / "checkpoints" / "generator_best.h5")

    # Build + load generator
    log("Building generator...")
    G = build_and_load_generator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        weights_path=weights_path,
        lr=LR,
        beta_1=BETA_1,
    )

    # Decide class counts
    if args.per_class:
        class_counts = parse_per_class(args.per_class)
        selected = sorted(class_counts.keys())
    else:
        selected = parse_classes(args.classes) if args.classes else list(range(NUM_CLASSES))
        if args.samples_per_class is not None:
            class_counts = {c: int(args.samples_per_class) for c in selected}
        elif args.num_samples is not None:
            if not args.balanced and len(selected) == 1:
                class_counts = {selected[0]: int(args.num_samples)}
            elif args.balanced:
                per = int(math.floor(args.num_samples / len(selected)))
                if per <= 0:
                    raise ValueError("num-samples too small for the number of selected classes.")
                class_counts = {c: per for c in selected}
            else:
                raise ValueError("When using --num-samples, you must pass --balanced or a single class via --classes.")
        else:
            # default: 1000 per class across all classes
            class_counts = {c: 1000 for c in selected}
            log("No counts provided; defaulting to 1000 per class.")

    log(f"Sampling plan: {class_counts}")

    # Generate
    x_synth, y_synth = generate_for_classes(
        G=G,
        latent_dim=LATENT_DIM,
        img_shape=IMG_SHAPE,
        class_counts=class_counts,
        num_classes=NUM_CLASSES,
        save_dir=samples_dir,
        save_per_class_npy=not args.no_save_per_class,
        truncation=args.truncation,
    )
    log(f"Generated synthetic arrays: x {x_synth.shape}, y {y_synth.shape}")

    # Save combined arrays for convenience
    np.save(samples_dir / "x_synth.npy", x_synth)
    np.save(samples_dir / "y_synth.npy", y_synth)

    # Optional grid
    if args.grid:
        rows, cols = args.grid
        n = min(rows * cols, x_synth.shape[0])
        titles = None
        # optional: annotate with class ids in the grid
        try:
            y_int = np.argmax(y_synth, axis=1)
            titles = [str(y_int[i]) for i in range(n)]
        except Exception:
            pass
        grid_path = Path(args.grid_path) if args.grid_path else (samples_dir / "preview_grid.png")
        save_grid(x_synth[:n], IMG_SHAPE, rows, cols, grid_path, titles=titles)
        log(f"Saved preview grid to {grid_path}")

    # Metadata for reproducibility
    meta = {
        "timestamp": now_ts(),
        "img_shape": IMG_SHAPE,
        "num_classes": NUM_CLASSES,
        "latent_dim": LATENT_DIM,
        "weights": str(weights_path),
        "classes": sorted(list(class_counts.keys())),
        "class_counts": class_counts,
        "seed": args.seed,
        "truncation": args.truncation,
        "outdir": str(samples_dir),
        "x_synth_path": str(samples_dir / "x_synth.npy"),
        "y_synth_path": str(samples_dir / "y_synth.npy"),
    }
    with open(samples_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Wrote metadata.json to {samples_dir}")

    log("Done.")
    return 0

def save_grid_from_generator(
    generator: tf.keras.Model,
    num_classes: int,
    latent_dim: int,
    *,
    n: int = 9,
    path: Optional[Path | str] = None,
    per_class: bool = True,
    seed: int = 42,
):
    """
    Generate a 1 x n preview grid from a conditional generator and save to disk.

    - Assumes generator outputs [-1, 1]; rescales to [0, 1].
    - Reuses the existing `save_grid(images01, img_shape, rows, cols, out_path, titles=None)`
      defined earlier in this file.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))

    # sample noise
    z = rng.normal(0.0, 1.0, size=(n, latent_dim)).astype(np.float32)

    # choose labels
    if per_class:
        labels = np.arange(n) % num_classes
    else:
        labels = rng.integers(low=0, high=num_classes, size=n)
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    # generate images [-1, 1] -> [0, 1]
    imgs = generator.predict([z, y], verbose=0)
    imgs01 = np.clip((imgs + 1.0) / 2.0, 0.0, 1.0)

    out_path = Path(path) if path is not None else Path("preview.png")
    # reuse your existing grid saver
    save_grid(
        images01=imgs01,
        img_shape=imgs01.shape[1:],  # (H, W, C)
        rows=1,
        cols=n,
        out_path=out_path,
        titles=[f"class {int(l)}" for l in labels],
    )
    return out_path

if __name__ == "__main__":
    sys.exit(main())
