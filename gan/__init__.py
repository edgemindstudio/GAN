# gan/__init__.py

"""
Public API for the GAN package.

This module keeps imports light so `import gan` is fast and side-effect free.
It exposes:
- MODEL_NAME: canonical name used across logs/artifacts.
- __version__: best-effort package version.
- ConditionalDCGANPipeline: lazily imported pipeline class.
- get_pipeline(config_path=None): convenience factory.

Note:
- Do NOT run this file directly. Use `python app/main.py` or `python -m gan.pipeline`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

MODEL_NAME = "GAN"

# Best-effort version resolution (works if installed as a package)
try:
    __version__ = _pkg_version("gan")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


def __getattr__(name: str):
    """
    Lazy attribute access so importing `gan` does not import heavy deps
    until they're actually needed.
    """
    if name == "ConditionalDCGANPipeline":
        # Import only when accessed
        from .pipeline import ConditionalDCGANPipeline
        return ConditionalDCGANPipeline
    if name == "get_pipeline":
        # Small factory so callers can do: from gan import get_pipeline
        def _factory(config_path: str | None = None):
            from .pipeline import ConditionalDCGANPipeline
            return ConditionalDCGANPipeline(config_path=config_path)
        return _factory
    raise AttributeError(f"module 'gan' has no attribute {name!r}")


__all__ = [
    "MODEL_NAME",
    "__version__",
    "ConditionalDCGANPipeline",
    "get_pipeline",
]

if __name__ == "__main__":
    # Friendly message if someone accidentally runs this file
    import sys
    print("gan/__init__.py is a package initializer, not an entrypoint.\n"
          "Run `python app/main.py` or `python -m gan.pipeline` instead.")
    sys.exit(1)
