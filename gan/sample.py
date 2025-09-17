# gan/sample.py
"""
Sampling utilities + unified-CLI synth entrypoint for the Conditional DCGAN.

What this provides
------------------
- Rebuild generator via `gan.models.build_models(...)` and (optionally) load weights.
- Generate synthetic samples either via:
    * Unified Orchestrator: `synth(cfg, output_root, seed)` → PNGs + manifest, or
    * (optionally) small preview via `save_grid_from_generator(...)`.
- Saves per-class PNGs in {output_root}/{class}/{seed}/gan_XXXXX.png.

Conventions
-----------
- Generator outputs tanh in [-1, 1]; we rescale to [0, 1] for PNGs.
- Config keys supported (with fallbacks): IMG_SHAPE, NUM_CLASSES, LATENT_DIM, LR, BETA_1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from gan.models import build_models


# -----------------------------
# Small helpers
# -----------------------------
def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


def _latents(n: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)


# -----------------------------
# Checkpoint loading
# -----------------------------
def load_g_from_checkpoints(
    ckpt_dir: Path | str,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    latent_dim: int,
    lr: float = 2e-4,
    beta_1: float = 0.5,
) -> tf.keras.Model:
    """
    Instantiate the conditional generator and load weights from G_best→G_last.
    If no compatible checkpoint exists or loading fails, returns random-initialized G.
    """
    ckpt_dir = Path(ckpt_dir)
    models = build_models(latent_dim=latent_dim, num_classes=num_classes, img_shape=img_shape, lr=lr, beta_1=beta_1)
    G: tf.keras.Model = models["generator"]

    # Build weights by a dummy forward pass (helps some Keras versions)
    H, W, C = img_shape
    dummy_z = tf.zeros((1, latent_dim), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    _ = G([dummy_z, dummy_y], training=False)

    best = ckpt_dir / "G_best.weights.h5"
    last = ckpt_dir / "G_last.weights.h5"
    to_load = best if best.exists() else last

    if not to_load.exists():
        print(f"[ckpt][warn] no GAN generator checkpoint in {ckpt_dir}; using random weights.")
        return G

    try:
        G.load_weights(str(to_load))
        print(f"[ckpt] Loaded {to_load.name}")
    except Exception as e:
        print(f"[ckpt][warn] failed to load {to_load.name}: {e}\n→ continuing with random weights.")
    return G


# -----------------------------
# Core generation
# -----------------------------
def _generate_batch_01(
    G: tf.keras.Model,
    *,
    class_id: int,
    count: int,
    latent_dim: int,
    num_classes: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Generate `count` images for a single class id. Returns (count, H, W, C) in [0,1].
    """
    z = _latents(count, latent_dim, seed=seed)
    y = tf.keras.utils.to_categorical(np.full((count,), class_id), num_classes=num_classes).astype(np.float32)
    g = G.predict([z, y], verbose=0)               # [-1, 1] from tanh
    g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)       # → [0, 1]
    return g01.astype(np.float32, copy=False)


# -----------------------------
# Public helper: quick preview grid
# -----------------------------
def save_grid_from_generator(
    generator: tf.keras.Model,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    latent_dim: int,
    n_per_class: int = 1,
    path: Path | str = "artifacts/gan/summaries/preview.png",
    seed: Optional[int] = 42,
) -> Path:
    """
    Generate a class-conditioned preview grid and save to disk.

    Layout: rows = classes (0..K-1), cols = n_per_class
    """
    H, W, C = img_shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    imgs = []
    for k in range(num_classes):
        z = rng.normal(0.0, 1.0, size=(n_per_class, latent_dim)).astype(np.float32)
        y = tf.keras.utils.to_categorical(np.full((n_per_class,), k), num_classes=num_classes).astype(np.float32)
        g = generator.predict([z, y], verbose=0)          # [-1, 1]
        g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)          # [0, 1]
        imgs.append(g01)
    imgs = np.concatenate(imgs, axis=0)                   # (K*n_per_class, H, W, C)

    rows, cols = num_classes, n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            img = imgs[idx]; idx += 1
            if C == 1:
                ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"C{r}", rotation=0, labelpad=10, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# -----------------------------
# Unified Orchestrator entrypoint
# -----------------------------
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Generate S PNGs/class into {output_root}/{class}/{seed}/... and return a manifest.

    Expects (optionally) generator checkpoints at:
      artifacts/gan/checkpoints/{G_best,G_last}.weights.h5
    """
    # Resolve shapes & counts (support legacy spellings)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    LATENT_DIM = int(_cfg_get(cfg, "LATENT_DIM", _cfg_get(cfg, "gan.latent_dim", 100)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))
    LR = float(_cfg_get(cfg, "LR", _cfg_get(cfg, "gan.lr", 2e-4)))
    BETA_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "gan.beta_1", 0.5)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.gan_checkpoints",
                             artifacts_root / "gan" / "checkpoints"))

    # Seeds for reproducibility
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))

    # Build + (optionally) load generator
    G = load_g_from_checkpoints(
        ckpt_dir,
        img_shape=(H, W, C),
        num_classes=K,
        latent_dim=LATENT_DIM,
        lr=LR,
        beta_1=BETA_1,
    )

    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    # Generate per class
    for k in range(K):
        imgs01 = _generate_batch_01(
            G,
            class_id=k,
            count=S,
            latent_dim=LATENT_DIM,
            num_classes=K,
            seed=seed,
        )  # (S, H, W, C)

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(S):
            p = cls_dir / f"gan_{j:05d}.png"
            _save_png(imgs01[j], p)
            paths.append({"path": str(p), "label": int(k)})
        per_class_counts[str(k)] = int(S)

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
    }
    return manifest


__all__ = [
    "load_g_from_checkpoints",
    "save_grid_from_generator",
    "synth",
]
