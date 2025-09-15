# common/data.py

"""
Dataset utilities shared across model repos.

Supports:
- Loading USTC-TFC2016 malware images from .npy files
- Reshaping to (H, W, C) with C=1 if needed
- Normalization to [0,1] or [-1,1] (GAN-friendly)
- One-hot encoding of labels
- Creating train/val/test numpy arrays
- Building tf.data.Datasets with shuffle/cache/prefetch
- Basic dataset stats printing

This module intentionally has no model-specific imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import tensorflow as tf


# -------------------------------
# Types & config
# -------------------------------

Array = np.ndarray
NormMode = Literal["zero_one", "minus_one_one"]  # [0,1] vs [-1,1]


@dataclass(frozen=True)
class DatasetArrays:
    """Numpy arrays for the full split."""
    x_train: Array
    y_train: Array  # one-hot
    x_val: Array
    y_val: Array    # one-hot
    x_test: Array
    y_test: Array   # one-hot


@dataclass(frozen=True)
class DatasetPipelines:
    """tf.data pipelines for the full split."""
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset


# -------------------------------
# Core loaders
# -------------------------------

def load_numpy_dataset(
    data_dir: str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
    normalize: NormMode = "minus_one_one",
    seed: int = 42,
    verbose: bool = True,
) -> DatasetArrays:
    """
    Load USTC-TFC2016 arrays and return train/val/test numpy arrays.

    Expects the following files in `data_dir`:
      - train_data.npy, train_labels.npy
      - test_data.npy,  test_labels.npy

    Args:
        data_dir: Folder containing the .npy files.
        img_shape: Target image shape (H, W, C). C is typically 1.
        num_classes: Number of label classes.
        val_fraction: Fraction of the provided *test* set to use as validation.
        normalize: 'minus_one_one' for GANs ([-1,1]) or 'zero_one' ([0,1]).
        seed: RNG seed for reproducible splitting/shuffling.
        verbose: Print basic dataset stats.

    Returns:
        DatasetArrays with one-hot labels and images shaped to (N, H, W, C).
    """
    _assert_required_files(data_dir)

    # Load
    x_train = np.load(os.path.join(data_dir, "train_data.npy"))
    y_train = np.load(os.path.join(data_dir, "train_labels.npy"))
    x_test  = np.load(os.path.join(data_dir, "test_data.npy"))
    y_test  = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Reshape images
    x_train = _ensure_img_shape(x_train, img_shape)
    x_test  = _ensure_img_shape(x_test,  img_shape)

    # Normalize
    x_train = _normalize(x_train, mode=normalize)
    x_test  = _normalize(x_test,  mode=normalize)

    # One-hot labels (accept int or already-one-hot)
    y_train_oh = _ensure_one_hot(y_train, num_classes)
    y_test_oh  = _ensure_one_hot(y_test,  num_classes)

    # Split test -> val/test
    rng = np.random.default_rng(seed)
    idx = np.arange(x_test.shape[0])
    rng.shuffle(idx)

    split = int(len(idx) * val_fraction)
    val_idx, test_idx = idx[:split], idx[split:]

    x_val,  y_val  = x_test[val_idx],  y_test_oh[val_idx]
    x_test, y_test = x_test[test_idx], y_test_oh[test_idx]

    if verbose:
        _print_basic_stats(x_train, y_train_oh, x_val, y_val, x_test, y_test)

    return DatasetArrays(
        x_train=x_train, y_train=y_train_oh,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
    )


# -------------------------------
# tf.data builders
# -------------------------------

def build_tf_pipelines(
    arrays: DatasetArrays,
    batch_size: int,
    shuffle_buffer: int = 8192,
    cache: bool = True,
    drop_remainder: bool = False,
    prefetch: int | None = tf.data.AUTOTUNE,
    seed: int = 42,
) -> DatasetPipelines:
    """
    Create performant tf.data pipelines from numpy arrays.

    Args:
        arrays: Output of `load_numpy_dataset`.
        batch_size: Global batch size.
        shuffle_buffer: Buffer size for shuffling training data.
        cache: Cache datasets in memory (recommended).
        drop_remainder: Drop last partial batch (disable for eval).
        prefetch: Prefetch setting; defaults to AUTOTUNE.
        seed: Seed for deterministic shuffle.

    Returns:
        DatasetPipelines for train/val/test.
    """
    def _make(x, y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        if cache:
            ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch is not None:
            ds = ds.prefetch(prefetch)
        return ds

    train = _make(arrays.x_train, arrays.y_train, training=True)
    val   = _make(arrays.x_val,   arrays.y_val,   training=False)
    test  = _make(arrays.x_test,  arrays.y_test,  training=False)

    return DatasetPipelines(train=train, val=val, test=test)


# -------------------------------
# Helpers
# -------------------------------

def _assert_required_files(data_dir: str) -> None:
    req = [
        "train_data.npy", "train_labels.npy",
        "test_data.npy",  "test_labels.npy",
    ]
    missing = [f for f in req if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in '{data_dir}': {missing}\n"
            "Expected: train_data.npy, train_labels.npy, test_data.npy, test_labels.npy"
        )


def _ensure_img_shape(x: Array, img_shape: Tuple[int, int, int]) -> Array:
    """
    Accepts (N, H, W), (N, H, W, C) or (N, D=H*W*C). Returns (N, H, W, C) float32.
    """
    H, W, C = img_shape
    x = np.asarray(x)
    if x.ndim == 2:  # (N, D)
        if x.shape[1] != H * W * C:
            raise ValueError(f"Flat dimension {x.shape[1]} does not match H*W*C={H*W*C}.")
        x = x.reshape((-1, H, W, C))
    elif x.ndim == 3:  # (N, H, W) -> add channel
        x = x[..., None]
        if C != 1:
            raise ValueError(f"Input has no channel axis but img_shape C={C} != 1.")
    elif x.ndim == 4:  # (N, H, W, C)
        # If channels mismatch but is 1, try to squeeze/expand intelligently
        if x.shape[1:4] != (H, W, C):
            try:
                x = x.reshape((-1, H, W, C))
            except Exception as e:
                raise ValueError(f"Cannot reshape input of shape {x.shape} to {(H,W,C)}") from e
    else:
        raise ValueError(f"Unsupported image array ndim={x.ndim}")
    return x.astype(np.float32, copy=False)


def _normalize(x: Array, mode: NormMode) -> Array:
    """
    Normalize raw inputs to [0,1] or [-1,1].

    - If raw data appears scaled like [0,255], we divide by 255 first.
    - If raw is already [0,1] we keep it.
    - If raw is [-1,1] and mode is minus_one_one, we keep it.
    """
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())

    # Bring to [0,1] baseline
    if x_max > 1.5:         # very likely [0,255]
        x = x / 255.0
    elif x_min < 0.0:       # already [-1,1] or similar
        x = (x + 1.0) / 2.0

    if mode == "minus_one_one":
        x = (x * 2.0) - 1.0
    elif mode == "zero_one":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    return np.clip(x, -1.0 if mode == "minus_one_one" else 0.0, 1.0)


def _ensure_one_hot(y: Array, num_classes: int) -> Array:
    """
    Accepts int labels shape (N,) or one-hot shape (N, K). Returns one-hot (N, K).
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        # already one-hot
        return y.astype(np.float32, copy=False)
    if y.ndim != 1:
        raise ValueError(f"Labels must be (N,) int or (N,{num_classes}) one-hot. Got {y.shape}")
    y_int = y.astype(int)
    if y_int.min() < 0 or y_int.max() >= num_classes:
        raise ValueError(f"Label values must be in [0,{num_classes-1}]. Found range [{y_int.min()},{y_int.max()}].")
    return np.eye(num_classes, dtype=np.float32)[y_int]


def _print_basic_stats(
    x_train: Array, y_train: Array,
    x_val: Array,   y_val: Array,
    x_test: Array,  y_test: Array,
) -> None:
    def counts(y_oh: Array) -> Dict[int, int]:
        c = y_oh.argmax(1)
        uniq, cnt = np.unique(c, return_counts=True)
        return {int(k): int(v) for k, v in zip(uniq, cnt)}

    H, W, C = x_train.shape[1:]
    print("\n[common.data] Dataset summary")
    print(f"  Image shape: (H,W,C)=({H},{W},{C})")
    print(f"  Train: {x_train.shape[0]}  | Val: {x_val.shape[0]}  | Test: {x_test.shape[0]}")
    print(f"  Train class counts: {counts(y_train)}")
    print(f"  Val   class counts: {counts(y_val)}")
    print(f"  Test  class counts: {counts(y_test)}\n")


# -------------------------------
# Convenience entrypoint
# -------------------------------

def load_arrays_and_pipelines(
    data_dir: str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    batch_size: int,
    val_fraction: float = 0.5,
    normalize: NormMode = "minus_one_one",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[DatasetArrays, DatasetPipelines]:
    """
    One-call convenience to get both numpy arrays and tf.data pipelines.
    """
    arrays = load_numpy_dataset(
        data_dir=data_dir,
        img_shape=img_shape,
        num_classes=num_classes,
        val_fraction=val_fraction,
        normalize=normalize,
        seed=seed,
        verbose=verbose,
    )
    pipes = build_tf_pipelines(arrays, batch_size=batch_size, seed=seed)
    return arrays, pipes
