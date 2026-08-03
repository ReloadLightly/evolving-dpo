#!/usr/bin/env python3
"""Phase 0/1 runner: benchmark one registered loss (or all of them).

Examples
--------
Hermetic plumbing check — no network at all, a few seconds:

    python scripts/run_baseline.py --offline --grid

CPU sanity check against a real (~1 MB) hub model, ~1 minute:

    python scripts/run_baseline.py --dry-run

Real baseline on a free Colab T4 (see docs/colab_quickstart.md):

    python scripts/run_baseline.py --loss dpo
    python scripts/run_baseline.py --loss simpo      # reference-free, ~2x faster
    python scripts/run_baseline.py --grid            # all registered losses

Each loss runs at its own published hyperparameters (see LOSS_DEFAULTS in
losses.py) unless you override them on the command line — SimPO's beta lives
on a different scale from DPO's, so a shared beta would be a rigged fight.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from evolving_dpo import (  # noqa: E402
    DRY_RUN, OFFLINE_RUN, EvalConfig, LOSS_REGISTRY, LOSS_DEFAULTS, run_candidate,
)


def apply_loss_defaults(cfg: EvalConfig, name: str, overrides: dict) -> EvalConfig:
    """Set each loss's own beta/extra kwargs, unless the user overrode them."""
    defaults = LOSS_DEFAULTS.get(name, {})
    train = cfg.train
    if "beta" in defaults and overrides.get("beta") is None:
        train = dataclasses.replace(train, beta=defaults["beta"])
    if defaults.get("extra"):
        train = dataclasses.replace(
            train, extra_loss_kwargs={**train.extra_loss_kwargs, **defaults["extra"]}
        )
    return dataclasses.replace(cfg, train=train)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", default="dpo", choices=sorted(LOSS_REGISTRY))
    p.add_argument("--grid", action="store_true", help="run every registered loss")
    p.add_argument("--dry-run", action="store_true",
                   help="tiny hub model + synthetic data on CPU: plumbing only")
    p.add_argument("--offline", action="store_true",
                   help="like --dry-run but hermetic: no network whatsoever")
    p.add_argument("--model", default=None, help="override model id")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--displacement-penalty", type=float, default=None,
                   help="weight on likelihood displacement in the fitness "
                        "(default 0.0: measured, not optimized against)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    base = OFFLINE_RUN if args.offline else (DRY_RUN if args.dry_run else EvalConfig())
    if args.model:
        base = dataclasses.replace(base, model_id=args.model)
    if args.n_train:
        base = dataclasses.replace(base, n_train=args.n_train)
    if args.n_eval:
        base = dataclasses.replace(base, n_eval=args.n_eval)
    if args.displacement_penalty is not None:
        base = dataclasses.replace(base, displacement_penalty=args.displacement_penalty)
    for fname, val in (("steps", args.steps), ("beta", args.beta), ("lr", args.lr)):
        if val is not None:
            base = dataclasses.replace(
                base, train=dataclasses.replace(base.train, **{fname: val})
            )

    losses = sorted(LOSS_REGISTRY) if args.grid else [args.loss]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.offline or args.dry_run:
        print("NOTE: smoke-test mode. The synthetic pairs are trivially\n"
              "separable, so pref_accuracy ~ 1.0 for every loss. This checks\n"
              "that the plumbing works; it is NOT a result. Real numbers come\n"
              "from a hub model on UltraFeedback (docs/colab_quickstart.md).")

    results = {}
    for name in losses:
        cfg = apply_loss_defaults(base, name, {"beta": args.beta})
        print(f"\n=== {name} (beta={cfg.train.beta}, "
              f"extra={cfg.train.extra_loss_kwargs}) ===")
        m = run_candidate(name, cfg)
        results[name] = m
        keys = ("combined_score", "pref_accuracy", "mean_margin",
                "drift_per_token", "chosen_logp_shift", "displacement_rate",
                "used_reference", "diverged", "seconds")
        print(json.dumps({k: m.get(k) for k in keys}, indent=2, default=str))

    tag = ("offline" if args.offline
           else "dryrun" if args.dry_run
           else base.model_id.split("/")[-1])
    out = outdir / f"baseline_{tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
