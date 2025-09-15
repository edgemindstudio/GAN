# gan/pipeline.py

"""
Conditional DCGAN pipeline (training + synthesis) used by app/main.py.

- Robust to missing cfg keys (sane defaults).
- Keras 3-compliant weight filenames: *.weights.h5 (required by save_weights()).
- Balanced per-class synthesis saved to .npy.
- Optional log callback cfg["LOG_CB"](epoch, d_loss|(d_loss,d_acc), g_loss).

Defaults if absent in cfg:
  IMG_SHAPE=(40,40,1), NUM_CLASSES=9, LATENT_DIM=100,
  EPOCHS=2000, BATCH_SIZE=256, LR=2e-4, BETA_1=0.5,
  NOISE_AFTER=200, LOG_EVERY≈EPOCHS/40 (min 50),
  SAMPLES_PER_CLASS=1000,
  ARTIFACTS.checkpoints="artifacts/gan/checkpoints",
  ARTIFACTS.synthetic="artifacts/gan/synthetic"
"""

from __future__ import annotations

import math
import time
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf

from gan.models import build_models  # must compile and return {"generator","discriminator","gan"}


# ---------------------------- Small helpers ----------------------------

def _to_scalar(x) -> float:
    """Normalize Keras/TensorFlow outputs (scalar, 0-D array/tensor, [loss, acc], etc.) to float."""
    if isinstance(x, (list, tuple)):
        return _to_scalar(x[0])
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _loss_and_acc(batch_out) -> Tuple[float, Optional[float]]:
    """Parse model.train_on_batch(...) return -> (loss, acc_or_None)."""
    if isinstance(batch_out, (list, tuple)):
        loss = _to_scalar(batch_out[0])
        acc = _to_scalar(batch_out[1]) if len(batch_out) > 1 else None
        return loss, acc
    return _to_scalar(batch_out), None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------- Pipeline ----------------------------

class ConditionalDCGANPipeline:
    """Training + synthesis orchestration for conditional DCGAN."""

    DEFAULTS = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "LATENT_DIM": 100,
        "EPOCHS": 2000,
        "BATCH_SIZE": 256,
        "LR": 2e-4,
        "BETA_1": 0.5,
        "NOISE_AFTER": 200,
        "SAMPLES_PER_CLASS": 1000,
        "ARTIFACTS": {
            "checkpoints": "artifacts/gan/checkpoints",
            "synthetic": "artifacts/gan/synthetic",
        },
    }

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}

        d = self.DEFAULTS
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))
        self.latent_dim: int = int(self.cfg.get("LATENT_DIM", d["LATENT_DIM"]))
        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))
        self.noise_after: int = int(self.cfg.get("NOISE_AFTER", d["NOISE_AFTER"]))
        # ~40 logs over full run (but at least one early save)
        self.log_every: int = int(self.cfg.get("LOG_EVERY", max(50, self.epochs // 40)))
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))

        arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
        self.ckpt_dir = Path(arts.get("checkpoints", d["ARTIFACTS"]["checkpoints"]))
        self.synth_dir = Path(arts.get("synthetic", d["ARTIFACTS"]["synthetic"]))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional logging callback: cb(epoch:int, d_loss or (d_loss,d_acc), g_loss)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Build fresh models (compiled inside build_models)
        models_dict = build_models(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            img_shape=self.img_shape,
            lr=self.lr,
            beta_1=self.beta_1,
        )
        self.G: tf.keras.Model = models_dict["generator"]
        self.D: tf.keras.Model = models_dict["discriminator"]
        self.GAN: tf.keras.Model = models_dict["gan"]

    # -------- optional small print logger --------
    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")

    # ---------------------------- Training ----------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Args:
            x_train: images in [-1, 1], shape (N,H,W,C)
            y_train: one-hot labels, shape (N,num_classes)
        Returns:
            (generator, discriminator)
        """
        H, W, C = self.img_shape
        assert x_train.shape[1:] == (H, W, C), f"Expected x_train shape (*,{H},{W},{C}), got {x_train.shape}"
        assert y_train.shape[1] == self.num_classes, "y_train must be one-hot with num_classes columns"

        steps_per_epoch = max(1, math.ceil(len(x_train) / self.batch_size))
        best_g_loss = float("inf")

        for epoch in range(self.epochs):
            perm = np.random.permutation(len(x_train))
            d_losses, g_losses = [], []
            d_acc_epoch = None  # keep the last batch acc if available

            for step in range(steps_per_epoch):
                sl = slice(step * self.batch_size, (step + 1) * self.batch_size)
                idx = perm[sl]
                real_imgs = x_train[idx].astype(np.float32)
                real_lbls = y_train[idx].astype(np.float32)

                n = real_imgs.shape[0]
                z = np.random.normal(0, 1, (n, self.latent_dim)).astype(np.float32)
                fake_cls = np.random.randint(0, self.num_classes, size=(n, 1))
                fake_lbls = tf.keras.utils.to_categorical(fake_cls, self.num_classes).astype(np.float32)

                gen_imgs = self.G.predict([z, fake_lbls], verbose=0)

                # small Gaussian noise after warmup (stabilization trick)
                if epoch + 1 > self.noise_after:
                    real_imgs = real_imgs + np.random.normal(0, 0.01, real_imgs.shape)
                    gen_imgs = gen_imgs + np.random.normal(0, 0.01, gen_imgs.shape)

                # label smoothing & noisy labels
                real_y = np.random.uniform(0.9, 1.0, size=(n, 1)).astype(np.float32)
                fake_y = np.random.uniform(0.0, 0.1, size=(n, 1)).astype(np.float32)

                # --- D step ---
                self.D.trainable = True
                d_out_real = self.D.train_on_batch([real_imgs, real_lbls], real_y)
                d_out_fake = self.D.train_on_batch([gen_imgs, fake_lbls], fake_y)
                d_real_loss, d_real_acc = _loss_and_acc(d_out_real)
                d_fake_loss, d_fake_acc = _loss_and_acc(d_out_fake)
                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                if (d_real_acc is not None) and (d_fake_acc is not None):
                    d_acc_epoch = 0.5 * (d_real_acc + d_fake_acc)
                d_losses.append(d_loss)

                # --- G step (want D(G(z,y)) -> 1) ---
                self.D.trainable = False
                g_out = self.GAN.train_on_batch([z, fake_lbls], np.ones((n, 1), dtype=np.float32))
                g_loss = _to_scalar(g_out)
                g_losses.append(g_loss)

            d_mean = float(np.mean(d_losses)) if d_losses else float("nan")
            g_mean = float(np.mean(g_losses)) if g_losses else float("nan")

            # user-provided logging callback (prints + tensorboard in app/main.py)
            if self.log_cb:
                self.log_cb(epoch + 1, (d_mean, d_acc_epoch) if d_acc_epoch is not None else d_mean, g_mean)

            # ---- Save periodic epoch weights (Keras 3 needs *.weights.h5) ----
            if (epoch + 1) % self.log_every == 0 or epoch == 0:
                self.G.save_weights(str(self.ckpt_dir / f"G_epoch_{epoch+1:04d}.weights.h5"))

            # track & save best generator by mean loss
            if g_mean < best_g_loss:
                best_g_loss = g_mean
                self.G.save_weights(str(self.ckpt_dir / "G_best.weights.h5"))

        # always save a final "last" checkpoint
        self.G.save_weights(str(self.ckpt_dir / "G_last.weights.h5"))
        return self.G, self.D

    # ---------------------------- Synthesis ----------------------------

    def synthesize(self, G: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced per-class synthetic dataset and save it under
        ARTIFACTS.synthetic. Returns (x_synth in [0,1], y_synth one-hot).

        If a generator `G` is passed, it will be used. Otherwise this method
        builds a generator and loads the latest available checkpoint.

        Saves:
          - gen_class_<k>.npy         (per-class images in [0,1])
          - labels_class_<k>.npy      (int labels for that class)
          - x_synth.npy, y_synth.npy  (concatenated convenience dumps)
        """
        H, W, C = self.img_shape
        per_class = int(self.cfg.get("SAMPLES_PER_CLASS", 1000))

        # Build+load generator if not provided
        if G is None:
            md = build_models(
                latent_dim=self.latent_dim,
                num_classes=self.num_classes,
                img_shape=self.img_shape,
                lr=self.lr,
                beta_1=self.beta_1,
            )
            G = md["generator"]

            # prefer *.weights.h5; fall back to legacy *.h5
            ckpt_patterns = [
                str(self.ckpt_dir / "G_epoch_*.weights.h5"),
                str(self.ckpt_dir / "G_best.weights.h5"),
                str(self.ckpt_dir / "G_last.weights.h5"),
                str(self.ckpt_dir / "G_epoch_*.h5"),
                str(self.ckpt_dir / "G_best.h5"),
                str(self.ckpt_dir / "G_last.h5"),
            ]
            candidates: list[str] = []
            for pat in ckpt_patterns:
                candidates.extend(glob.glob(pat))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                G.load_weights(latest)
                self._log(f"Loaded generator weights: {os.path.basename(latest)}")
            else:
                self._log("WARNING: No generator weights found; generating with an untrained generator.")

        # Generate per-class
        xs, ys = [], []
        _ensure_dir(self.synth_dir)

        for cls in range(self.num_classes):
            z = np.random.normal(0, 1, (per_class, self.latent_dim)).astype(np.float32)
            y = tf.keras.utils.to_categorical(np.full((per_class, 1), cls), self.num_classes).astype(np.float32)

            g = G.predict([z, y], verbose=0)           # [-1, 1]
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)   # -> [0, 1]

            xs.append(g01.reshape(-1, H, W, C))
            ys.append(y)

            # per-class dumps for traceability
            np.save(self.synth_dir / f"gen_class_{cls}.npy", g01)
            np.save(self.synth_dir / f"labels_class_{cls}.npy", np.full((per_class,), cls, dtype=np.int32))

        x_synth = np.concatenate(xs, axis=0)
        y_synth = np.concatenate(ys, axis=0)

        # convenience combined dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        self._log(f"Synthesized {x_synth.shape[0]} samples ({per_class} per class) -> {self.synth_dir}")
        return x_synth, y_synth
