"""evolving-dpo: preference-optimization objectives, and the peer-review study.

Torch-dependent names (the trainer, the losses, the evaluator) are resolved
lazily via PEP 562, so the data-side modules import on a machine with no torch
installed:

    from evolving_dpo import review_pairs, surface_baseline   # no torch needed
    from evolving_dpo import run_candidate                    # imports torch

Building and analyzing the dataset should not require a 2 GB download.
"""

from typing import Any

# Torch-free: safe to import eagerly.
from . import review_pairs, surface_baseline

_LAZY = {
    "LOSS_REGISTRY": ".losses",
    "LOSS_DEFAULTS": ".losses",
    "get_loss": ".losses",
    "needs_reference": ".losses",
    "EvalConfig": ".evaluate",
    "DRY_RUN": ".evaluate",
    "OFFLINE_RUN": ".evaluate",
    "run_candidate": ".evaluate",
}

__all__ = ["review_pairs", "surface_baseline", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    """Import torch-backed names only when they are actually used."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
