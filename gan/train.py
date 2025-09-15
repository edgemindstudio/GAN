# gan/train.py

"""
Train a Conditional DCGAN on the USTC-TFC2016 (or compatible) dataset.

Key features
------------
- Loads config.yaml (IMG_SHAPE, NUM_CLASSES, LATENT_DIM, EPOCHS, BATCH_SIZE, LR, BETA_1, DATA_PATH).
- Builds G/D from gan.models.build_models() to ensure a single source of truth.
- GAN training loop with:
    * label smoothing & optional Gaussian noise after warmup
    * per-epoch (or every N epochs) FID against a validation split
    * TensorBoard logging
    * periodic and "best-FID" checkpoints
    * optional preview grids
- Optional post-training sampling (balanced per-class) to artifacts/samples/.

Usage
-----
# vanilla training with config.yaml
python -m gan.train

# override a few options
python -m gan.train --epochs 1000 --eval-every 10 --save-every 50 --grid 4 9

# resume generator/discriminator weights
python -m gan.train --g-weights artifacts/checkpoints/generator_last.h5 \
                    --d-weights artifacts/checkpoints/discriminator_last.h5

# generate 1000 samples per class after training
python -m gan.train --sample-after --samples-per-class 1000
"""

from __future__ import annotations

import os
import math
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers, models

# Local modules
from gan.models import build_models

# Optional: use the same FID implementation as eval_common
try:
    from eval_common import fid_keras as compute_fid_01
except Exception:
    compute_fid_01 = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now_ts()}] {msg}")


def make_dirs(root: Path):
    (root / "artifacts" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "samples").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)


def save_grid(images01: np.ndarray, img_shape: tuple[int, int, int], rows: int, cols: int, out_path: Path):
    """Save a grid from images in [0,1]."""
    import matplotlib.pyplot as plt
    H, W, C = img_shape
    n = min(rows * cols, images01.shape[0])
    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im = images01[i].reshape(H, W, C)
        if C == 1:
            plt.imshow(im.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(im, 0.0, 1.0))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_numpy_splits(data_path: Path, img_shape: tuple[int, int, int]):
    """
    Loads train/test .npy and splits test in half => val/test.
    Returns:
        x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh
        with images in [0,1], shapes (N,H,W,1), labels one-hot.
    """
    H, W, C = img_shape
    x_train = np.load(data_path / "train_data.npy")
    y_train = np.load(data_path / "train_labels.npy")
    x_test = np.load(data_path / "test_data.npy")
    y_test = np.load(data_path / "test_labels.npy")

    # to [0,1]
    x_train = (x_train.astype(np.float32)) / 255.0
    x_test = (x_test.astype(np.float32)) / 255.0

    # reshape (N,H,W,1)
    x_train = x_train.reshape((-1, H, W, C))
    x_test = x_test.reshape((-1, H, W, C))

    # split test into val/test
    split = len(x_test) // 2
    x_val01, y_val = x_test[:split], y_test[:split]
    x_test01, y_test = x_test[split:], y_test[split:]

    # to one-hot
    num_classes = int(np.max(y_train)) + 1
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh


# ---------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------
def scale_to_neg1_1(x01: np.ndarray) -> np.ndarray:
    return (x01 * 2.0) - 1.0


def maybe_add_noise(x: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return x
    return x + np.random.normal(0.0, std, size=x.shape).astype(np.float32)


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
def train(
    cfg_path: Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_every: int = 25,
    save_every: int = 50,
    label_smooth: tuple[float, float] = (0.9, 1.0),  # for real labels
    fake_label_range: tuple[float, float] = (0.0, 0.1),
    noise_after: int = 200,     # start adding noise to images after N epochs
    noise_std: float = 0.01,
    grid: tuple[int, int] | None = None,
    g_weights: Path | None = None,
    d_weights: Path | None = None,
    sample_after: bool = False,
    samples_per_class: int = 0,
    seed: int = 42,
):
    # --- setup ---
    set_seeds(seed)
    enable_gpu_memory_growth()

    with open(cfg_path, "r") as f:
        cfg = json.loads(json.dumps(__import__("yaml").safe_load(f)))  # robust load

    IMG_SHAPE   = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    NUM_CLASSES = int(cfg.get("NUM_CLASSES", 9))
    LATENT_DIM  = int(cfg.get("LATENT_DIM", 100))
    EPOCHS      = int(epochs if epochs is not None else cfg.get("EPOCHS", 5000))
    BATCH_SIZE  = int(batch_size if batch_size is not None else cfg.get("BATCH_SIZE", 256))
    LR          = float(cfg.get("LR", 2e-4))
    BETA_1      = float(cfg.get("BETA_1", 0.5))
    DATA_PATH   = Path(cfg.get("DATA_PATH", Path(cfg_path).resolve().parents[0] / "USTC-TFC2016_malware"))

    root = Path(__file__).resolve().parents[1]
    make_dirs(root)
    tb_dir = root / "logs" / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(str(tb_dir))
    ckpt_dir = root / "artifacts" / "checkpoints"
    samples_dir = root / "artifacts" / "samples"

    log(f"Config: IMG_SHAPE={IMG_SHAPE}, NUM_CLASSES={NUM_CLASSES}, LATENT_DIM={LATENT_DIM}, "
        f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, BETA_1={BETA_1}")
    log(f"DATA_PATH={DATA_PATH}")

    # --- data ---
    x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh = load_numpy_splits(DATA_PATH, IMG_SHAPE)
    x_train = scale_to_neg1_1(x_train01)  # GAN expects [-1,1]

    # --- models ---
    models_dict = build_models(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        lr=LR,
        beta_1=BETA_1,
    )
    G: tf.keras.Model = models_dict["generator"]
    D: tf.keras.Model = models_dict["discriminator"]

    # allow resume
    if g_weights and Path(g_weights).exists():
        G.load_weights(str(g_weights))
        log(f"Loaded generator weights from {g_weights}")
    if d_weights and Path(d_weights).exists():
        D.load_weights(str(d_weights))
        log(f"Loaded discriminator weights from {d_weights}")

    # standalone D compile (binary crossentropy)
    D.compile(optimizer=Adam(LR, BETA_1), loss="binary_crossentropy", metrics=["accuracy"])

    # Combined model (G->D), D frozen
    noise_in = layers.Input(shape=(LATENT_DIM,), name="z")
    label_in = layers.Input(shape=(NUM_CLASSES,), name="y_onehot")
    fake_img = G([noise_in, label_in])
    D.trainable = False
    valid = D([fake_img, label_in])
    combined = models.Model([noise_in, label_in], valid, name="G_over_D")
    combined.compile(optimizer=Adam(LR, BETA_1), loss="binary_crossentropy")

    # --- training loop ---
    steps_per_epoch = math.ceil(x_train.shape[0] / BATCH_SIZE)
    best_fid = float("inf")

    log(f"Start training for {EPOCHS} epochs ({steps_per_epoch} steps/epoch). TensorBoard -> {tb_dir}")

    for epoch in range(1, EPOCHS + 1):
        # shuffle indices for this epoch
        idx = np.random.permutation(x_train.shape[0])

        d_losses, g_losses = [], []

        for step in range(steps_per_epoch):
            # ---- prepare real batch ----
            batch_idx = idx[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            real_imgs = x_train[batch_idx]
            real_lbls = y_train_oh[batch_idx]

            # optional noise after warmup
            if epoch > noise_after and noise_std > 0:
                real_imgs = maybe_add_noise(real_imgs, noise_std)

            # ---- generate fake batch ----
            z = np.random.normal(0.0, 1.0, size=(real_imgs.shape[0], LATENT_DIM)).astype(np.float32)
            fake_class_int = np.random.randint(0, NUM_CLASSES, size=(real_imgs.shape[0], 1))
            fake_lbls = tf.keras.utils.to_categorical(fake_class_int, NUM_CLASSES).astype(np.float32)
            gen_imgs = G.predict([z, fake_lbls], verbose=0)

            if epoch > noise_after and noise_std > 0:
                gen_imgs = maybe_add_noise(gen_imgs, noise_std)

            # ---- labels with smoothing ----
            real_y = np.random.uniform(label_smooth[0], label_smooth[1], size=(real_imgs.shape[0], 1)).astype(np.float32)
            fake_y = np.random.uniform(fake_label_range[0], fake_label_range[1], size=(gen_imgs.shape[0], 1)).astype(np.float32)

            # ---- train D (real + fake) ----
            D.trainable = True
            d_loss_real = D.train_on_batch([real_imgs, real_lbls], real_y)
            d_loss_fake = D.train_on_batch([gen_imgs, fake_lbls], fake_y)

            # Keras returns list [loss, acc]
            if isinstance(d_loss_real, list) and isinstance(d_loss_fake, list):
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])
            else:
                d_loss = 0.5 * (float(d_loss_real) + float(d_loss_fake))
                d_acc = np.nan

            # ---- train G (mislead D) ----
            D.trainable = False
            z = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, LATENT_DIM)).astype(np.float32)
            g_lbls_int = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE, 1))
            g_lbls = tf.keras.utils.to_categorical(g_lbls_int, NUM_CLASSES).astype(np.float32)
            g_loss = combined.train_on_batch([z, g_lbls], np.ones((BATCH_SIZE, 1), dtype=np.float32))

            d_losses.append(d_loss)
            g_losses.append(g_loss if np.isscalar(g_loss) else g_loss[0])

        # --- epoch end: logs ---
        d_loss_ep = float(np.mean(d_losses))
        g_loss_ep = float(np.mean(g_losses))

        # periodic preview grid
        if grid is not None:
            rows, cols = grid
            z = np.random.normal(0.0, 1.0, size=(rows * cols, LATENT_DIM)).astype(np.float32)
            # cycle labels 0..NUM_CLASSES-1
            cyc = np.arange(rows * cols) % NUM_CLASSES
            y_cyc = tf.keras.utils.to_categorical(cyc, NUM_CLASSES).astype(np.float32)
            g = G.predict([z, y_cyc], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            save_grid(g01, IMG_SHAPE, rows, cols, root / "artifacts" / "samples" / f"grid_epoch_{epoch:04d}.png")

        # periodic FID (lower is better)
        fid_val = None
        if (compute_fid_01 is not None) and (epoch % max(1, eval_every) == 0):
            # compare against a capped/equalized validation slice
            n_fid = min(200, x_val01.shape[0])
            real01 = x_val01[:n_fid]
            z = np.random.normal(0.0, 1.0, size=(n_fid, LATENT_DIM)).astype(np.float32)
            labels_int = np.random.randint(0, NUM_CLASSES, size=(n_fid,))
            y_oh = tf.keras.utils.to_categorical(labels_int, NUM_CLASSES).astype(np.float32)
            g = G.predict([z, y_oh], verbose=0)
            fake01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            fid_val = float(compute_fid_01(real01, fake01))
            # checkpoint best by FID
            if fid_val < best_fid:
                best_fid = fid_val
                G.save_weights(str(ckpt_dir / "generator_best.h5"))
                D.save_weights(str(ckpt_dir / "discriminator_best.h5"))
                with open(ckpt_dir / "best_fid.json", "w") as f:
                    json.dump({"epoch": epoch, "best_fid": best_fid, "timestamp": now_ts()}, f)
                log(f"[BEST] Epoch {epoch} new best FID={best_fid:.4f} -> saved *_best.h5")

        # periodic save (last)
        if epoch % max(1, save_every) == 0 or epoch == EPOCHS:
            G.save_weights(str(ckpt_dir / "generator_last.h5"))
            D.save_weights(str(ckpt_dir / "discriminator_last.h5"))

        # console log
        if fid_val is not None:
            log(f"Epoch {epoch:04d} | D_loss {d_loss_ep:.4f} | G_loss {g_loss_ep:.4f} | FID {fid_val:.4f}")
        else:
            log(f"Epoch {epoch:04d} | D_loss {d_loss_ep:.4f} | G_loss {g_loss_ep:.4f}")

        # tensorboard
        with writer.as_default():
            tf.summary.scalar("loss/D", d_loss_ep, step=epoch)
            tf.summary.scalar("loss/G", g_loss_ep, step=epoch)
            if fid_val is not None:
                tf.summary.scalar("FID/val", fid_val, step=epoch)
        writer.flush()

    log("Training complete.")
    # Save final
    G.save_weights(str(ckpt_dir / "generator_final.h5"))
    D.save_weights(str(ckpt_dir / "discriminator_final.h5"))

    # Optional: post-training sampling (balanced per class)
    if sample_after and samples_per_class > 0:
        log(f"Sampling {samples_per_class} per class to {samples_dir} ...")
        for k in range(NUM_CLASSES):
            z = np.random.normal(0.0, 1.0, size=(samples_per_class, LATENT_DIM)).astype(np.float32)
            y = tf.keras.utils.to_categorical(np.full((samples_per_class,), k), NUM_CLASSES).astype(np.float32)
            g = G.predict([z, y], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            np.save(samples_dir / f"gen_class_{k}.npy", g01)
            np.save(samples_dir / f"labels_class_{k}.npy", np.full((samples_per_class,), k, dtype=np.int32))
        log("Sampling done.")

    return 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="Path to config.yaml")

    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate FID every N epochs")
    parser.add_argument("--save-every", type=int, default=50, help="Save last checkpoints every N epochs")

    parser.add_argument("--label-smooth", type=float, nargs=2, default=(0.9, 1.0),
                        help="Real label smoothing range [low high]")
    parser.add_argument("--fake-label-range", type=float, nargs=2, default=(0.0, 0.1),
                        help="Fake label range [low high]")
    parser.add_argument("--noise-after", type=int, default=200, help="Start noise injection after this epoch")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std for images after warmup")

    parser.add_argument("--grid", type=int, nargs=2, default=None, help="Save preview grid: ROWS COLS per epoch")

    parser.add_argument("--g-weights", type=str, default=None, help="Path to resume generator weights")
    parser.add_argument("--d-weights", type=str, default=None, help="Path to resume discriminator weights")

    parser.add_argument("--sample-after", action="store_true", help="Generate per-class samples after training")
    parser.add_argument("--samples-per-class", type=int, default=0, help="Samples per class if --sample-after")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    return train(
        cfg_path=Path(args.config),
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        label_smooth=tuple(args.label_smooth),
        fake_label_range=tuple(args.fake_label_range),
        noise_after=args.noise_after,
        noise_std=args.noise_std,
        grid=tuple(args.grid) if args.grid else None,
        g_weights=Path(args.g_weights) if args.g_weights else None,
        d_weights=Path(args.d_weights) if args.d_weights else None,
        sample_after=args.sample_after,
        samples_per_class=int(args.samples_per_class),
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
