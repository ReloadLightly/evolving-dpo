#!/usr/bin/env python3
"""Phase 0/1 runner: benchmark one registered loss (or all of them).

Examples
--------
CPU sanity check, no downloads beyond a ~1 MB test model, ~1 minute:

    python scripts/run_baseline.py --dry-run

Real baseline on a free Colab T4 (see docs/colab_quickstart.md):

    python scripts/run_baseline.py --loss dpo
    python scripts/run_baseline.py --grid          # all registered losses
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from evolving_dpo import DRY_RUN, EvalConfig, LOSS_REGISTRY, run_candidate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", default="dpo", choices=sorted(LOSS_REGISTRY))
    p.add_argument("--grid", action="store_true", help="run every registered loss")
    p.add_argument("--dry-run", action="store_true",
                   help="tiny model + synthetic data on CPU: checks plumbing only")
    p.add_argument("--model", default=None, help="override model id")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = DRY_RUN if args.dry_run else EvalConfig()
    if args.model:
        cfg = dataclasses.replace(cfg, model_id=args.model)
    if args.n_train:
        cfg = dataclasses.replace(cfg, n_train=args.n_train)
    if args.n_eval:
        cfg = dataclasses.replace(cfg, n_eval=args.n_eval)
    for field, val in (("steps", args.steps), ("beta", args.beta), ("lr", args.lr)):
        if val is not None:
            cfg = dataclasses.replace(
                cfg, train=dataclasses.replace(cfg.train, **{field: val})
            )

    losses = sorted(LOSS_REGISTRY) if args.grid else [args.loss]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in losses:
        print(f"\n=== {name} ===")
        m = run_candidate(name, cfg)
        results[name] = m
        keys = ("combined_score", "pref_accuracy", "mean_margin",
                "drift_per_token", "diverged", "seconds")
        print(json.dumps({k: m.get(k) for k in keys}, indent=2, default=str))

    tag = "dryrun" if args.dry_run else cfg.model_id.split("/")[-1]
    out = outdir / f"baseline_{tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
