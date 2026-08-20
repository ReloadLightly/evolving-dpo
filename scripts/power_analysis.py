#!/usr/bin/env python3
"""Task 8: how big an effect can this design actually detect?

Run BEFORE training. The point is to find out now, while the design is still
changeable, whether the planned comparisons can detect a plausible effect --
particularly the H3 interaction, which needs far more data than a main effect
and is the paper's contribution.

    python scripts/power_analysis.py                       # projected counts
    python scripts/power_analysis.py --index data/pairs_index.csv   # real ones

Writes results/power.md.

Three analyses, because the preregistration implies three different tests and
they do not have the same power:

1. H1, pair-level. Every arm is scored on the same held-out pairs, so the
   comparison is paired (McNemar), not two independent samples. Paired is
   substantially more powerful, and the difference is worth knowing.

2. H1, seed-level. The preregistration's actual falsification criterion is
   "non-overlapping 95% CIs across 5 seeds". That is a test on five numbers,
   not on thousands of pairs, so between-seed variance -- not dataset size --
   is what limits it. It is also *stricter* than p < 0.05: two means whose 95%
   CIs merely touch are separated at roughly p = 0.006. Reported here so the
   criterion is chosen deliberately rather than by habit.

3. H3, the interaction. A 2x2 of tuning area against test area. The variance
   of an interaction contrast is the sum of the variances of its four cells,
   so its standard error is about twice that of a main effect -- meaning
   roughly 4x the data for the same detectable size.

Normal approximations throughout, which is standard for proportions at these
sample sizes and honest about being an approximation.
"""

from __future__ import annotations

import argparse
import csv
import collections
import math
import pathlib
import sys

Z_ALPHA_2 = 1.959963985   # two-sided alpha = 0.05
Z_POWER_80 = 0.841621234  # 80% power
T_CRIT_4DF = 2.776445105  # 95% CI on a mean of 5 seeds

# Measured in data/YIELD_PROJECTION.md by running the construction rule over
# real ICLR ratings and outcomes from papercopilot/paperlists.
PROJECTED = {
    2020: {"pairs": 1125, "accepted": 379, "against_majority": 299},
    2021: {"pairs": 1613, "accepted": 362, "against_majority": 327},
    2022: {"pairs": 1553, "accepted": 502, "against_majority": 236},
    2023: {"pairs": 2220, "accepted": 704, "against_majority": 374},
    2024: {"pairs": 3488, "accepted": 1101, "against_majority": 573},
}
TRAIN_YEARS = (2020, 2021, 2022, 2023)
TEST_YEARS = (2024,)


def mde_unpaired(n_per_group: int, baseline: float = 0.5) -> float:
    """Detectable difference between two independent proportions."""
    if n_per_group <= 0:
        return float("nan")
    lo, hi = 1e-6, 0.5
    for _ in range(200):
        d = (lo + hi) / 2
        p1, p2 = baseline, min(0.999, baseline + d)
        pbar = (p1 + p2) / 2
        need = ((Z_ALPHA_2 * math.sqrt(2 * pbar * (1 - pbar))
                 + Z_POWER_80 * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / d) ** 2
        if need > n_per_group:
            lo = d
        else:
            hi = d
    return hi


def mde_paired(n_pairs: int, discordance: float) -> float:
    """Detectable difference when both arms score the SAME pairs (McNemar).

    `discordance` is the share of pairs the two arms disagree on. It is
    unknown before running, so the caller sweeps a range of plausible values.
    """
    if n_pairs <= 0:
        return float("nan")
    lo, hi = 1e-6, min(0.5, discordance)
    for _ in range(200):
        d = (lo + hi) / 2
        inner = discordance - d * d
        if inner <= 0:
            hi = d
            continue
        need = ((Z_ALPHA_2 * math.sqrt(discordance)
                 + Z_POWER_80 * math.sqrt(inner)) / d) ** 2
        if need > n_pairs:
            lo = d
        else:
            hi = d
    return hi


def mde_seed_level(between_seed_sd: float, n_seeds: int = 5) -> dict:
    """Detectable gap under the preregistration's own criterion.

    Two arms, n_seeds runs each. Returns both the non-overlapping-95%-CI gap
    the preregistration specifies and, for comparison, an ordinary two-sample
    t-test at alpha = 0.05.
    """
    sem = between_seed_sd / math.sqrt(n_seeds)
    # CIs stop overlapping once the means differ by the sum of the two margins.
    non_overlap = 2 * T_CRIT_4DF * sem
    # Welch-ish two-sample t at 80% power, df = 2(n-1).
    t_test = (2.306 + 0.889) * between_seed_sd * math.sqrt(2.0 / n_seeds)
    return {"non_overlapping_ci": non_overlap, "two_sample_t": t_test,
            "sem": sem, "ratio": non_overlap / t_test if t_test else float("nan")}


def mde_interaction(n_per_cell: int, baseline: float = 0.60) -> float:
    """Detectable interaction in a 2x2, expressed in accuracy points.

    Works on the log-odds scale, where the interaction contrast is
    (b11 - b12) - (b21 - b22) and its variance is the sum of four cell
    variances, then converts back to a difference in proportions at the
    baseline. The conversion is local and therefore approximate.
    """
    if n_per_cell <= 0:
        return float("nan")
    var_cell = 1.0 / (n_per_cell * baseline * (1 - baseline))
    se_interaction = math.sqrt(4 * var_cell)
    log_odds = (Z_ALPHA_2 + Z_POWER_80) * se_interaction
    # Convert a log-odds contrast into accuracy points near the baseline.
    odds = baseline / (1 - baseline)
    upper = (odds * math.exp(log_odds / 2)) / (1 + odds * math.exp(log_odds / 2))
    lower = (odds * math.exp(-log_odds / 2)) / (1 + odds * math.exp(-log_odds / 2))
    return upper - lower


def n_for_interaction(target: float, baseline: float = 0.60) -> int:
    """Per-cell N needed to detect an interaction of `target` accuracy points."""
    for n in range(50, 400001, 50):
        if mde_interaction(n, baseline) <= target:
            return n
    return -1


def balanced_eval_size(pairs: int, accepted: int) -> int:
    """Size of a 50/50 accepted/rejected evaluation set (HANDOFF section 6).

    The minority class is the binding constraint, so the balanced set is twice
    the smaller class -- notably smaller than the raw pair count.
    """
    return 2 * min(accepted, pairs - accepted)


def load_index(path: pathlib.Path) -> dict:
    """Read real counts from data/pairs_index.csv if it exists."""
    per_year: dict[int, dict] = collections.defaultdict(
        lambda: {"pairs": 0, "accepted": 0, "against_majority": 0})
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            year = int(row["year"])
            per_year[year]["pairs"] += 1
            if str(row.get("accepted", "")).lower() in ("true", "1"):
                per_year[year]["accepted"] += 1
            if str(row.get("majority_agreed_with_decision", "")).lower() == "false":
                per_year[year]["against_majority"] += 1
    return dict(per_year)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default=None,
                   help="data/pairs_index.csv; falls back to projected counts")
    p.add_argument("--out", default="results/power.md")
    p.add_argument("--baseline", type=float, default=0.60,
                   help="assumed accuracy of the tuned arm (default 0.60)")
    p.add_argument("--areas", type=int, default=2, help="areas in the H3 split")
    args = p.parse_args()

    if args.index and pathlib.Path(args.index).exists():
        counts, source = load_index(pathlib.Path(args.index)), f"real counts from {args.index}"
    else:
        counts, source = PROJECTED, "projected counts (data/YIELD_PROJECTION.md)"
        if args.index:
            print(f"! {args.index} not found; using projections", file=sys.stderr)

    train = sum(counts[y]["pairs"] for y in TRAIN_YEARS if y in counts)
    test_raw = sum(counts[y]["pairs"] for y in TEST_YEARS if y in counts)
    test_acc = sum(counts[y]["accepted"] for y in TEST_YEARS if y in counts)
    test_bal = balanced_eval_size(test_raw, test_acc)
    against = sum(counts[y]["against_majority"] for y in TEST_YEARS if y in counts)
    # The against-majority subset shrinks with the balancing, proportionally.
    against_bal = round(against * test_bal / test_raw) if test_raw else 0

    L = []
    L.append("# Power analysis (Task 8)")
    L.append("")
    L.append("Generated by `scripts/power_analysis.py`. Do not edit by hand.")
    L.append("")
    L.append(f"Source of counts: {source}.")
    L.append("Two-sided alpha = 0.05, 80% power, normal approximations throughout.")
    L.append("")
    L.append("## The evaluation set is smaller than the pair count")
    L.append("")
    L.append(f"- Training pairs (2020-2023): **{train:,}**")
    L.append(f"- Test pairs (2024), raw: **{test_raw:,}**")
    L.append(f"- Test pairs after 50/50 accept/reject balancing: **{test_bal:,}**")
    L.append(f"- Of those, decided against the reviewer majority: **~{against_bal:,}**")
    L.append("")
    L.append("Balancing is not optional (HANDOFF section 6): without it, always")
    L.append("preferring the negative review scores ~70%. But the minority class binds,")
    L.append(f"so it costs {100 * (1 - test_bal / test_raw):.0f}% of the test set. Budget for the balanced number.")
    L.append("")

    L.append("## 1. H1, pair-level")
    L.append("")
    L.append(f"Detectable difference in pairwise accuracy on {test_bal:,} balanced test pairs:")
    L.append("")
    L.append("| Comparison | Assumption | Detectable difference |")
    L.append("|---|---|---|")
    L.append(f"| Two independent samples | baseline {args.baseline:.0%} | "
             f"{100 * mde_unpaired(test_bal, args.baseline):.2f} pp |")
    for psi in (0.15, 0.25, 0.35):
        L.append(f"| Paired (McNemar) | {psi:.0%} of pairs discordant | "
                 f"{100 * mde_paired(test_bal, psi):.2f} pp |")
    L.append("")
    L.append("Every arm scores the same pairs, so the paired row is the right one.")
    L.append("At these sizes a 2-3 point difference is detectable, which is well")
    L.append("below the gap H1 expects against an untuned base model.")
    L.append("")
    L.append(f"On the against-the-majority subset (~{against_bal:,} pairs), the honest test:")
    L.append("")
    L.append(f"- independent: **{100 * mde_unpaired(against_bal, args.baseline):.2f} pp**")
    L.append(f"- paired at 25% discordance: **{100 * mde_paired(against_bal, 0.25):.2f} pp**")
    L.append("")
    L.append("Smaller, but still usable -- this subset is a genuine test rather than")
    L.append("a footnote, which is worth knowing before it gets cut for space.")
    L.append("")

    L.append("## 2. H1, seed-level -- the criterion the preregistration actually states")
    L.append("")
    L.append("\"Non-overlapping 95% CIs across 5 seeds\" is a test on 5 numbers per arm.")
    L.append("Dataset size does not enter; between-seed SD does.")
    L.append("")
    L.append("| Between-seed SD | Non-overlapping 95% CI needs | Two-sample t needs | Ratio |")
    L.append("|---|---|---|---|")
    for sd in (0.005, 0.010, 0.020, 0.030):
        m = mde_seed_level(sd)
        L.append(f"| {100 * sd:.1f} pp | **{100 * m['non_overlapping_ci']:.2f} pp** | "
                 f"{100 * m['two_sample_t']:.2f} pp | {m['ratio']:.2f}x |")
    L.append("")
    L.append("Two consequences worth deciding on deliberately:")
    L.append("")
    L.append("1. **Non-overlapping CIs is stricter than p < 0.05.** Two means whose 95%")
    L.append("   CIs merely touch are separated at about p = 0.006. The criterion is")
    L.append("   defensible and conservative, but it is not \"significance\", and it")
    L.append("   should be described as the stronger claim it is.")
    L.append("2. **With 5 seeds, between-seed SD dominates everything.** If seeds vary by")
    L.append("   2 pp, no amount of test data lets the criterion resolve a 3 pp effect.")
    L.append("   Measure the seed SD in the first arm you train; if it is above ~1 pp,")
    L.append("   raise the seed count before running all five arms.")
    L.append("")

    L.append("## 3. H3, the interaction -- the binding constraint")
    L.append("")
    L.append(f"A 2x2 of tuning area against test area. With {args.areas} areas, the")
    L.append("balanced test set splits into that many test sets, each scored by every")
    L.append("model -- so per-cell N is the size of one test *area*, and the")
    L.append("interaction contrast carries the variance of all four cells, giving it")
    L.append("roughly twice the standard error of a main effect.")
    L.append("")
    # Cell structure matters here. Both models are scored on the SAME test
    # pairs of a given area, so the two cells sharing a test area share their
    # observations: per-cell N is the size of one test AREA, not the balanced
    # set divided by four. (That sharing also makes the within-area contrast
    # paired, which this analysis does not exploit -- so the numbers below are
    # conservative.)
    per_cell = test_bal // args.areas if args.areas else 0
    L.append("| Per-cell N | Detectable interaction |")
    L.append("|---|---|")
    for n in (100, 200, 300, 500, 750, 1000, 1500, 2000):
        L.append(f"| {n:,} | {100 * mde_interaction(n, args.baseline):.2f} pp |")
    L.append("")
    L.append(f"Your split gives roughly **{per_cell:,} pairs per cell** "
             f"({test_bal:,} balanced test pairs split into {args.areas} test areas), "
             f"detecting an interaction of about "
             f"**{100 * mde_interaction(per_cell, args.baseline):.2f} pp**.")
    L.append("")
    L.append("For reference, the per-cell N needed for a given target:")
    L.append("")
    L.append("| Target interaction | Per-cell N | Balanced test pairs needed |")
    L.append("|---|---|---|")
    for target in (0.02, 0.03, 0.05, 0.08, 0.10):
        n = n_for_interaction(target, args.baseline)
        L.append(f"| {100 * target:.0f} pp | {n:,} | {n * args.areas:,} |")
    L.append("")

    all_pairs = sum(counts[y]["pairs"] for y in counts)
    all_acc = sum(counts[y]["accepted"] for y in counts)
    all_bal = balanced_eval_size(all_pairs, all_acc)
    # A topic split holds out areas, not years, so it can draw on every year.
    h3_cell_all = all_bal // args.areas if args.areas else 0
    L.append("### If the topic split uses every year instead of 2024 alone")
    L.append("")
    L.append("The temporal and topic splits are separate designs (HANDOFF section 5).")
    L.append("H3 holds out *areas*, not years, so it may draw on the whole corpus --")
    L.append("as long as the paper says plainly that H3 is not additionally held out")
    L.append("in time, and the temporal result is reported separately.")
    L.append("")
    L.append(f"- All pairs 2020-2024: **{all_pairs:,}**, balanced: **{all_bal:,}**")
    L.append(f"- Per test area ({args.areas} areas): **{h3_cell_all:,}**")
    L.append(f"- Detectable interaction: **{100 * mde_interaction(h3_cell_all, args.baseline):.2f} pp**"
             f" (vs {100 * mde_interaction(per_cell, args.baseline):.2f} pp on 2024 alone)")
    L.append("")
    L.append("## What this means for the design")
    L.append("")
    L.append("- **Use few, large areas.** Interaction power scales with the *smallest*")
    L.append("  cell, so splitting into many fine-grained areas is the fastest way to")
    L.append("  make H3 untestable. Two well-separated areas beat six precise ones.")
    L.append("- **SNOR has no `primary_area`.** The split must be built from `keywords`")
    L.append("  or by clustering the Specter embeddings SNOR ships. Build it to")
    L.append("  maximize cell size subject to the two areas being genuinely distinct,")
    L.append("  and fix the choice before training (HANDOFF section 5).")
    L.append("- **Add ICLR 2025.** It roughly doubles the test set, and the interaction")
    L.append("  is the analysis that needs it most.")
    L.append("- **Report the interaction with its CI even if null.** A null with a tight")
    L.append("  CI says DPO does not capture a documented difference; a null with a wide")
    L.append("  CI says nothing at all. Only the power analysis distinguishes them, and")
    L.append("  that distinction is most of H3's value either way.")
    L.append("")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")

    print(f"Training pairs 2020-2023: {train:,}")
    print(f"Test pairs 2024 raw {test_raw:,} -> balanced {test_bal:,} "
          f"(against-majority ~{against_bal:,})")
    print(f"H1 paired MDE @25% discordance: "
          f"{100 * mde_paired(test_bal, 0.25):.2f} pp")
    print(f"H3 interaction MDE at {per_cell:,}/cell: "
          f"{100 * mde_interaction(per_cell, args.baseline):.2f} pp")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
