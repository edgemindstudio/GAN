# app/main.py
# =============================================================================
# Conditional DCGAN — production entry point (Phase-2 parity with other repos)
#
# Commands
# --------
# python -m app.main train         # train GAN, save checkpoints
# python -m app.main synth         # load latest G checkpoint and synthesize per-class samples
# python -m app.main eval          # standardized evaluation with/without synthetic
# python -m app.main all           # train -> synth -> eval
#
# Phase-2 outputs
# ---------------
# - runs/console.txt               (clean human-readable block; via gcs_core or local fallback)
# - runs/summary.jsonl             (append-only one JSON per run; via gcs_core or local fallback)
# - artifacts/.../summaries/*.json (pretty JSON in Phase-2 schema for the aggregator)
#
# Notes
# -----
# • Training uses images in [-1, 1]; evaluation uses [0, 1].
# • Images are float32 NHWC; labels are one-hot.
# • Includes a robust writer shim that supports multiple gcs_core signatures and
#   falls back to a local writer to avoid API drift failures (fixes 'real_dirs' errors).
# =============================================================================

from __future__ import annotations

# --- Make repo-local packages importable (gan/, etc.) -------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import argparse
import json
from typing import Dict, Tuple, Optional, Any, Mapping, Union

import numpy as np
import tensorflow as tf
import yaml

# ------------------------------------------------------------------------------
# gcs_core (the “frozen core”) for synth discovery + evaluation/writing
# ------------------------------------------------------------------------------
from gcs_core.synth_loader import resolve_synth_dir, load_synth_any
from gcs_core.val_common import compute_all_metrics, write_summary_with_gcs_core

# ------------------------------------------------------------------------------
# DCGAN pipeline & helpers (repo-local)
# ------------------------------------------------------------------------------
from gan.pipeline import ConditionalDCGANPipeline          # type: ignore
from gan.models import build_generator                     # type: ignore
try:
    # Newer repos expose this helper
    from gan.sample import save_grid_from_generator as save_grid  # type: ignore
except Exception:  # pragma: no cover
    save_grid = None  # type: ignore


# =============================================================================
# GPU niceties (safe on CPU-only machines)
# =============================================================================
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Deterministic NumPy + TF RNG."""
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_yaml(path: Path) -> Dict:
    """Parse YAML at `path`."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict) -> None:
    """Create artifact directories present in cfg (idempotent)."""
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot (N, K) float32; pass-through when already one-hot."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32")
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype("float32")


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map [0,1] → [-1,1] for tanh-based generators."""
    return (x01 - 0.5) * 2.0


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from four .npy files:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns:
      x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h
    Where each x_* is float32 in [0,1], shaped (N,H,W,C) and labels are one-hot.
    Splits provided test -> (val, test) using `val_fraction`.
    """
    H, W, C = img_shape
    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    def to_01_hwc(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if x.max() > 1.5:  # handle 0..255
            x = x / 255.0
        x = x.reshape((-1, H, W, C))
        return np.clip(x, 0.0, 1.0)

    x_train01 = to_01_hwc(x_train)
    x_test01  = to_01_hwc(x_test)

    y_train1h = one_hot(y_train, num_classes)
    y_test1h  = one_hot(y_test,  num_classes)

    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]
    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


def latest_generator_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Pick the most appropriate generator checkpoint (newest first by preference):
      1) G_best.weights.h5
      2) G_last.weights.h5
      3) Latest G_epoch_*.weights.h5
      4) Legacy: newest G_epoch_*.h5 (no '.weights')
    """
    preferred = [ckpt_dir / "G_best.weights.h5", ckpt_dir / "G_last.weights.h5"]
    for p in preferred:
        if p.exists():
            return p
    epoch_ckpts = sorted(ckpt_dir.glob("G_epoch_*.weights.h5"))
    if epoch_ckpts:
        return max(epoch_ckpts, key=lambda pp: pp.stat().st_mtime)
    legacy = sorted(ckpt_dir.glob("G_epoch_*.h5"))
    return max(legacy, key=lambda pp: pp.stat().st_mtime) if legacy else None


def _sanitize_synth(
    x_s: Optional[np.ndarray],
    y_s: Optional[np.ndarray],
    img_shape: Tuple[int, int, int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Drop non-finite samples and clamp to [0,1]. Leave (None, None) unchanged.
    Ensures shape (N, H, W, C).
    """
    if x_s is None or y_s is None or x_s.size == 0:
        return None, None
    H, W, C = img_shape
    x_s = np.asarray(x_s, dtype=np.float32)
    y_s = np.asarray(y_s, dtype=np.float32)
    if x_s.ndim == 3:
        x_s = x_s.reshape((-1, H, W, C))
    mask = np.isfinite(x_s).all(axis=(1, 2, 3))
    if not mask.any():
        print("[warn] All synthetic samples were non-finite; evaluating REAL-only.")
        return None, None
    if not mask.all():
        print(f"[warn] Dropping {(~mask).sum()} non-finite synthetic samples.")
    x_s = np.clip(x_s[mask], 0.0, 1.0)
    y_s = y_s[mask]
    return x_s, y_s


# =============================================================================
# Logging callback (TensorBoard-friendly)
# =============================================================================
def make_log_cb(tboard_dir: Optional[Path]):
    """
    Build a lightweight logging callback:
      cb(epoch, d_loss_metrics, g_loss_scalar)
    Where d_loss_metrics can be (loss,) or (loss, acc).
    """
    writer = tf.summary.create_file_writer(str(tboard_dir)) if tboard_dir else None

    def cb(epoch: int, d_loss, g_loss):
        d_loss_val = float(d_loss[0]) if isinstance(d_loss, (list, tuple)) else float(d_loss)
        d_acc_val = (
            float(d_loss[1]) if isinstance(d_loss, (list, tuple)) and len(d_loss) > 1 else None
        )
        gl = float(g_loss)
        msg = f"[epoch {epoch:05d}] D_loss={d_loss_val:.4f}"
        if d_acc_val is not None:
            msg += f" D_acc={d_acc_val:.4f}"
        msg += f" | G_loss={gl:.4f}"
        print(msg)
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/discriminator", d_loss_val, step=epoch)
                if d_acc_val is not None:
                    tf.summary.scalar("acc/discriminator", d_acc_val, step=epoch)
                tf.summary.scalar("loss/generator", gl, step=epoch)
                writer.flush()

    return cb


# =============================================================================
# Phase-2 writer shim (supports multiple gcs_core signatures + local fallback)
# =============================================================================
PathLike = Union[str, Path]

def _build_real_dirs(data_dir: Path) -> Dict[str, str]:
    """Stable 'real_dirs' mapping used by newer writers."""
    return {
        "train": str(data_dir / "train_data.npy"),
        "val":   f"{data_dir}/(split of test_data.npy)",
        "test":  f"{data_dir}/(split of test_data.npy)",
    }


def _map_util_names(util_block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize utility metric names to a stable schema."""
    if not util_block:
        return {}
    bal = util_block.get("balanced_accuracy", util_block.get("bal_acc"))
    return {
        "accuracy":               util_block.get("accuracy"),
        "macro_f1":               util_block.get("macro_f1"),
        "balanced_accuracy":      bal,
        "macro_auprc":            util_block.get("macro_auprc"),
        "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
        "ece":                    util_block.get("ece"),
        "brier":                  util_block.get("brier"),
        "per_class":              util_block.get("per_class"),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None) else float(a - b)


def _build_phase2_record(
    *,
    model_name: str,
    seed: int,
    images_counts: Mapping[str, Optional[int]],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct Phase-2 aggregator record from metrics and counts."""
    util_R  = _map_util_names(metrics.get("real_only"))
    util_RS = _map_util_names(metrics.get("real_plus_synth"))
    deltas = {
        "accuracy":           _delta(util_RS.get("accuracy"),          util_R.get("accuracy")),
        "macro_f1":           _delta(util_RS.get("macro_f1"),          util_R.get("macro_f1")),
        "balanced_accuracy":  _delta(util_RS.get("balanced_accuracy"), util_R.get("balanced_accuracy")),
        "macro_auprc":        _delta(util_RS.get("macro_auprc"),       util_R.get("macro_auprc")),
        "recall_at_1pct_fpr": _delta(util_RS.get("recall_at_1pct_fpr"),util_R.get("recall_at_1pct_fpr")),
        "ece":                _delta(util_RS.get("ece"),               util_R.get("ece")),
        "brier":              _delta(util_RS.get("brier"),             util_R.get("brier")),
    }
    generative = {
        "fid":          metrics.get("fid_macro"),
        "fid_macro":    metrics.get("fid_macro"),
        "cfid_macro":   metrics.get("cfid_macro"),
        "js":           metrics.get("js"),
        "kl":           metrics.get("kl"),
        "diversity":    metrics.get("diversity"),
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }
    return {
        "model": model_name,
        "seed":  int(seed),
        "images": {
            "train_real": int(images_counts.get("train_real") or 0),
            "val_real":   int(images_counts.get("val_real") or 0),
            "test_real":  int(images_counts.get("test_real") or 0),
            "synthetic":  (int(images_counts["synthetic"]) if images_counts.get("synthetic") is not None else None),
        },
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
    }


def _write_console_block(record: Dict[str, Any]) -> str:
    """Format a concise console block and return it."""
    gen = record.get("generative", {})
    util_R  = record.get("utility_real_only", {})
    util_RS = record.get("utility_real_plus_synth", {})
    counts  = record.get("images", {})
    lines = [
        f"Model: {record.get('model')}   Seed: {record.get('seed')}",
        f"Counts → train:{counts.get('train_real')}  "
        f"val:{counts.get('val_real')}  "
        f"test:{counts.get('test_real')}  "
        f"synth:{counts.get('synthetic')}",
        f"Generative → FID(macro): {gen.get('fid_macro')}  cFID(macro): {gen.get('cfid_macro')}  "
        f"JS: {gen.get('js')}  KL: {gen.get('kl')}  Div: {gen.get('diversity')}",
        f"Utility R   → acc: {util_R.get('accuracy')}  bal_acc: {util_R.get('balanced_accuracy')}  "
        f"macro_f1: {util_R.get('macro_f1')}",
        f"Utility R+S → acc: {util_RS.get('accuracy')}  bal_acc: {util_RS.get('balanced_accuracy')}  "
        f"macro_f1: {util_RS.get('macro_f1')}",
    ]
    return "\n".join(lines) + "\n"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def _save_console(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(block)


def _ensure_images_block(record: Dict[str, Any], images_counts: Mapping[str, Optional[int]]) -> None:
    """Ensure 'images' counts exist even if the core writer omitted them."""
    record.setdefault("images", {})
    for k, v in images_counts.items():
        if record["images"].get(k) is None:
            record["images"][k] = v


def _try_core_writer(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: Union[str, Path],
    output_console_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """
    Try multiple signatures of gcs_core.write_summary_with_gcs_core.
    Returns the record on success; None if every attempt fails.
    """
    base_kwargs = dict(
        model_name=model_name,
        seed=seed,
        fid_cap_per_class=fid_cap_per_class,
        output_json=str(output_json_path),
        output_console=str(output_console_path),
        metrics=dict(metrics),
        notes=notes,
    )
    real_dirs = _build_real_dirs(data_dir)

    attempts = [
        # Newest: real_dirs + images_counts + synth_dir
        dict(base_kwargs, real_dirs=real_dirs, images_counts=dict(images_counts), synth_dir=synth_dir),
        # Mid: real_dirs + synth_dir (no images_counts)
        dict(base_kwargs, real_dirs=real_dirs, synth_dir=synth_dir),
        # Oldest: synth_dir only
        dict(base_kwargs, synth_dir=synth_dir),
    ]

    for kw in attempts:
        try:
            rec = write_summary_with_gcs_core(**kw)
            _ensure_images_block(rec, images_counts)
            return rec
        except TypeError:
            continue  # signature mismatch – try next layout
        except Exception:
            continue  # internal failure – try next layout
    return None


def _local_write_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: Union[str, Path],
    output_console_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Build the Phase-2 record locally and write console + JSONL outputs.
    Used when gcs_core writer signatures don't match (API drift).
    """
    record = _build_phase2_record(
        model_name=model_name,
        seed=seed,
        images_counts=images_counts,
        metrics=metrics,
    )
    # Include a few extras for traceability:
    record.setdefault("meta", {})
    record["meta"].update({
        "notes": notes,
        "fid_cap_per_class": int(fid_cap_per_class),
        "synth_dir": synth_dir,
        "real_dirs": _build_real_dirs(data_dir),
    })

    console_block = _write_console_block(record)
    _save_console(Path(output_console_path), console_block)
    _append_jsonl(Path(output_json_path), record)
    return record


def _write_phase2_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: Optional[str],
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str = "",
    output_json_path: Union[str, Path] = "runs/summary.jsonl",
    output_console_path: Union[str, Path] = "runs/console.txt",
) -> Dict[str, Any]:
    """
    Version-agnostic writer:
      1) Try multiple gcs_core signatures.
      2) If all fail, write locally with a schema-compatible record.
    """
    sdir = synth_dir or ""
    rec = _try_core_writer(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )
    if rec is not None:
        return rec

    return _local_write_summary(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )


# =============================================================================
# Orchestration
# =============================================================================
def _build_generator_for_ckpt(cfg: Dict):
    """
    Rebuild a generator compatible with checkpoints. Supports both:
      • build_generator(latent_dim=..., num_classes=..., img_shape=..., lr=..., beta_1=...)
      • build_generator()
    """
    try:
        return build_generator(
            latent_dim=int(cfg["LATENT_DIM"]),
            num_classes=int(cfg["NUM_CLASSES"]),
            img_shape=tuple(cfg["IMG_SHAPE"]),
            lr=float(cfg.get("LR", 2e-4)),
            beta_1=float(cfg.get("BETA_1", 0.5)),
        )
    except TypeError:
        return build_generator()


def run_train(cfg: Dict) -> None:
    """Train DCGAN and write a small preview grid."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # DCGAN expects [-1,1] during training
    x_train_m11 = to_minus1_1(x_train01)

    pipe = ConditionalDCGANPipeline(cfg | {"LOG_CB": make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))})
    G, _ = pipe.train(x_train=x_train_m11, y_train=y_train)

    # Save a compact generator preview grid (1×n cycling class labels)
    preview_path = Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png"
    try:
        if callable(save_grid):
            # Newer helper signature (kwargs)
            try:
                save_grid(
                    generator=G,
                    num_classes=num_classes,
                    latent_dim=int(cfg["LATENT_DIM"]),
                    n=min(9, num_classes),
                    path=preview_path,
                    conditional=True,
                )
            except TypeError:
                # Legacy helper signature (positional)
                save_grid(G, num_classes, int(cfg["LATENT_DIM"]), n=min(9, num_classes), path=preview_path)  # type: ignore
        print(f"Saved preview grid to {preview_path}")
    except Exception:
        # Keep training usable even if preview fails
        pass


def run_synth(cfg: Dict) -> None:
    """Load latest generator checkpoint and synthesize balanced per-class arrays."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    ckpt_dir  = Path(cfg["ARTIFACTS"]["checkpoints"])
    synth_dir = Path(cfg["ARTIFACTS"]["synthetic"])

    ckpt = latest_generator_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No generator checkpoints found in {ckpt_dir}")

    # Rebuild generator and load weights
    G = _build_generator_for_ckpt(cfg)
    G.load_weights(str(ckpt))
    print(f"Loaded generator checkpoint: {ckpt.name}")

    pipe = ConditionalDCGANPipeline(cfg)
    x_s, y_s = pipe.synthesize(G)  # pipeline is responsible for saving arrays
    print(f"Synthesized: {x_s.shape[0]} samples (saved under {synth_dir}).")


def run_eval(cfg: Dict, include_synth: bool) -> None:
    """
    Phase-2 standardized evaluation:
      • Generative quality (FID / cFID / JS / KL / Diversity) on VAL vs SYNTH
      • Downstream utility on REAL test with the fixed small CNN
      • Outputs:
          - runs/console.txt
          - runs/summary.jsonl
          - artifacts/gan/summaries/ConditionalDCGAN_eval_summary_seed{SEED}.json
    """
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)
    Path("runs").mkdir(exist_ok=True)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])
    seed        = int(cfg.get("SEED", 42))
    fid_cap     = int(cfg.get("FID_CAP", 200))
    eval_epochs = int(cfg.get("EVAL_EPOCHS", 20))

    # --- REAL data ---
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # --- Optional SYNTH ---
    x_s, y_s = (None, None)
    synth_dir_str = ""
    if include_synth:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            synth_dir = resolve_synth_dir(cfg, repo_root)  # supports absolute/relative layouts
            x_s, y_s = load_synth_any(synth_dir, num_classes)  # monolithic or per-class dumps
            x_s, y_s = _sanitize_synth(x_s, y_s, img_shape)
            synth_dir_str = str(synth_dir)
            if x_s is not None:
                print(f"[eval] Using synthetic from {synth_dir} (N={len(x_s)})")
            else:
                print(f"[eval] WARN: no usable synthetic under {synth_dir}; evaluating REAL-only.")
        except Exception as e:
            print(f"[eval] WARN: could not load synthetic -> {e}. Evaluating REAL-only.")
            x_s, y_s = None, None
            synth_dir_str = str(Path(cfg.get("ARTIFACTS", {}).get("synthetic", "artifacts/synthetic")))
    else:
        synth_dir_str = str(Path(cfg.get("ARTIFACTS", {}).get("synthetic", "")))

    # --- Compute metrics (compute_all_metrics standardizes ranges/shapes) ---
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_tr, y_train_real=y_tr,
        x_val_real=x_va,   y_val_real=y_va,
        x_test_real=x_te,  y_test_real=y_te,
        x_synth=x_s,       y_synth=y_s,
        fid_cap_per_class=fid_cap,
        seed=seed,
        domain_embed_fn=None,
        epochs=eval_epochs,
    )

    # --- Version-agnostic writer (core shim -> local fallback) ---------------
    images_counts = {
        "train_real": int(x_tr.shape[0]),
        "val_real":   int(x_va.shape[0]),
        "test_real":  int(x_te.shape[0]),
        "synthetic":  (int(x_s.shape[0]) if isinstance(x_s, np.ndarray) else None),
    }

    record = _write_phase2_summary(
        model_name="ConditionalDCGAN",
        seed=seed,
        data_dir=data_dir,
        synth_dir=synth_dir_str,
        fid_cap_per_class=fid_cap,
        metrics=metrics,
        images_counts=images_counts,
        notes="phase2-real",
        output_json_path="runs/summary.jsonl",
        output_console_path="runs/console.txt",
    )

    # --- Pretty JSON copy under ARTIFACTS/summaries (authoritative for aggregator)
    out_path = Path(cfg["ARTIFACTS"]["summaries"]) / f"ConditionalDCGAN_eval_summary_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved evaluation summary to {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional DCGAN pipeline runner")
    p.add_argument("command", choices=["train", "synth", "eval", "all"], help="Which step to run")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    p.add_argument("--no-synth", action="store_true", help="(for eval/all) skip synthetic data in evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Sensible defaults (aligned with other repos)
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("FID_CAP", 200)
    cfg.setdefault("EVAL_EPOCHS", 20)     # evaluator CNN epochs (separate from GAN train)
    cfg.setdefault("LATENT_DIM", 128)
    cfg.setdefault("IMG_SHAPE", [40, 40, 1])
    cfg.setdefault("NUM_CLASSES", 9)
    cfg.setdefault("SAMPLES_PER_CLASS", 25)  # used by pipeline.synthesize
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/gan/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/gan/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/gan/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    # Attach training log callback for TensorBoard if available
    cfg["LOG_CB"] = make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))

    print(f"[config] Using {Path(args.config).resolve()}")
    print(f"Synth outputs -> {Path(cfg['ARTIFACTS']['synthetic']).resolve()}")

    if args.command == "train":
        run_train(cfg)
    elif args.command == "synth":
        run_synth(cfg)
    elif args.command == "eval":
        run_eval(cfg, include_synth=not args.no_synth)
    elif args.command == "all":
        run_train(cfg)
        run_synth(cfg)
        run_eval(cfg, include_synth=not args.no_synth)


if __name__ == "__main__":
    main()
