# common/interfaces.py

"""
Shared interfaces and config objects used by every model repo.

Goals
-----
- Provide consistent dataclasses for configs and artifact paths.
- Define function/class contracts so each model's `pipeline.py` plugs into the
  same CLI (app/main.py) and evaluator (eval/eval_common.py).
- Keep dependencies minimal (stdlib + typing).

Typical usage in a model repo (e.g., GAN):
------------------------------------------
from pathlib import Path
from common.interfaces import (Paths, TrainConfig, GenerateConfig, EvaluateConfig,
                               TrainResult, GenerateResult, IModelPipeline)

class GanPipeline(IModelPipeline):
    name = "GAN"

    def train(self, cfg: TrainConfig) -> TrainResult:
        # ... training code ...
        return TrainResult(best_checkpoint=cfg.paths.checkpoints / "gen_best.h5",
                           history={"loss": [..], "val_loss": [..]},
                           wall_time_sec=123.4)

    def generate(self, cfg: GenerateConfig) -> GenerateResult:
        # ... sampling code ...
        return GenerateResult(num_samples=cfg.total_samples or 0,
                              class_counts={0: 100, 1: 100},
                              output_dir=cfg.paths.synthetic)

    def evaluate(self, cfg: EvaluateConfig) -> dict:
        # ... call eval/eval_common.evaluate_model_suite(...)
        return {"model": self.name, ...}

# Or expose function-style APIs in pipeline.py:
#   def train_pipeline(cfg: TrainConfig) -> TrainResult: ...
#   def generate_pipeline(cfg: GenerateConfig) -> GenerateResult: ...
#   def evaluate_pipeline(cfg: EvaluateConfig) -> dict: ...
#
# These functions should honor the same contracts defined below.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Protocol, Tuple, runtime_checkable

# Reuse the normalization enum from common.data to avoid drift
try:
    # Local import; safe because this module has no TF/torch side effects.
    from .data import NormMode  # Literal["zero_one","minus_one_one"]
except Exception:
    # Fallback to avoid circular import during editors' static analysis
    from typing import Literal as NormMode  # type: ignore


# ---------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """
    Conventional paths used across repos.
    Create via `Paths.for_repo(root, model_name)` or by passing explicit paths.
    """
    repo_root: Path
    data_dir: Path
    artifacts_root: Path
    checkpoints: Path
    synthetic: Path
    summaries: Path
    tensorboard: Path
    logs: Path

    @staticmethod
    def for_repo(repo_root: Path, model_name: str, data_subdir: str = "USTC-TFC2016_malware") -> "Paths":
        """
        Build a consistent directory tree:
          repo_root/
            artifacts/<model_name>/{checkpoints,synthetic,summaries,tensorboard}
            logs/
            configs/
            app/
            ...
        """
        repo_root = repo_root.resolve()
        artifacts_root = repo_root / "artifacts" / model_name.lower()
        return Paths(
            repo_root=repo_root,
            data_dir=repo_root / data_subdir,
            artifacts_root=artifacts_root,
            checkpoints=artifacts_root / "checkpoints",
            synthetic=artifacts_root / "synthetic",
            summaries=artifacts_root / "summaries",
            tensorboard=artifacts_root / "tensorboard",
            logs=repo_root / "logs",
        )

    def ensure(self) -> "Paths":
        """Create folders if they do not exist; returns self for chaining."""
        for p in (self.artifacts_root, self.checkpoints, self.synthetic,
                  self.summaries, self.tensorboard, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------
# Config objects (training, generation, evaluation)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Model-agnostic training config."""
    img_shape: Tuple[int, int, int]           # (H, W, C), typically (40, 40, 1)
    num_classes: int
    batch_size: int
    epochs: int
    lr: float
    seed: int = 42
    beta1: Optional[float] = None             # for Adam-like optimizers (GAN, Diffusion)
    normalize: NormMode = "minus_one_one"     # GANs prefer [-1,1]; others may use [0,1]
    device: Optional[str] = None              # "cpu", "cuda:0", "mps", or None -> auto
    mixed_precision: bool = False
    gradient_clip_norm: Optional[float] = None
    paths: Paths = field(default_factory=lambda: Paths.for_repo(Path("."), "generic").ensure())
    extra: Dict[str, Any] = field(default_factory=dict)  # model-specific knobs

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in self.paths.__dict__.items()}
        return d


@dataclass(frozen=True)
class GenerateConfig:
    """Sampling / synthetic data generation config."""
    img_shape: Tuple[int, int, int]
    num_classes: int
    samples_per_class: Optional[int] = None   # if conditional generation
    total_samples: Optional[int] = None       # if unconditional or alternative cap
    seed: int = 42
    conditional: bool = True
    save_numpy: bool = True                   # save .npy arrays
    save_preview_grid: bool = True            # optionally save a PNG grid
    preview_grid_n: int = 16
    normalize_out: NormMode = "zero_one"      # generated arrays saved in [0,1] by default
    paths: Paths = field(default_factory=lambda: Paths.for_repo(Path("."), "generic").ensure())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in self.paths.__dict__.items()}
        return d


@dataclass(frozen=True)
class EvaluateConfig:
    """
    Evaluation config for utility & generative metrics.
    Used by eval/eval_common.evaluate_model_suite.
    """
    img_shape: Tuple[int, int, int]
    num_classes: int
    per_class_cap_for_fid: int = 200
    seed: int = 42
    paths: Paths = field(default_factory=lambda: Paths.for_repo(Path("."), "generic").ensure())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in self.paths.__dict__.items()}
        return d


# ---------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainResult:
    """Returned by pipeline.train()."""
    best_checkpoint: Optional[Path]
    history: Dict[str, List[float]] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "history": self.history,
            "wall_time_sec": self.wall_time_sec,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GenerateResult:
    """Returned by pipeline.generate()."""
    num_samples: int
    class_counts: Dict[int, int] = field(default_factory=dict)
    output_dir: Path = Path(".")
    preview_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "class_counts": {int(k): int(v) for k, v in self.class_counts.items()},
            "output_dir": str(self.output_dir),
            "preview_path": str(self.preview_path) if self.preview_path else None,
        }


# The evaluation summary is a plain dict produced by eval_common.evaluate_model_suite.
EvalSummary = Dict[str, Any]


# ---------------------------------------------------------------------
# Pipeline contracts (both class-based and function-based)
# ---------------------------------------------------------------------

class IModelPipeline:
    """
    Optional class-based interface you can implement in each repo.
    If you prefer function-based exports in pipeline.py, see the Protocols below.
    """
    name: str = "MODEL"

    def train(self, cfg: TrainConfig) -> TrainResult:            # pragma: no cover - interface
        raise NotImplementedError

    def generate(self, cfg: GenerateConfig) -> GenerateResult:   # pragma: no cover - interface
        raise NotImplementedError

    def evaluate(self, cfg: EvaluateConfig) -> EvalSummary:      # pragma: no cover - interface
        raise NotImplementedError


@runtime_checkable
class TrainFn(Protocol):
    def __call__(self, cfg: TrainConfig) -> TrainResult: ...


@runtime_checkable
class GenerateFn(Protocol):
    def __call__(self, cfg: GenerateConfig) -> GenerateResult: ...


@runtime_checkable
class EvaluateFn(Protocol):
    def __call__(self, cfg: EvaluateConfig) -> EvalSummary: ...


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def ensure_paths(paths: Paths) -> Paths:
    """Handy helper when a pipeline receives `cfg.paths` and wants to be safe."""
    return paths.ensure()


def human_size(n_bytes: int) -> str:
    """Pretty-print byte sizes."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f}{u}"
        size /= 1024.0
    return f"{size:.1f}PB"
