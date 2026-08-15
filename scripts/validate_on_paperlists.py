#!/usr/bin/env python3
"""Validate the pair-construction logic on real ICLR ratings, without OpenReview.

`papercopilot/paperlists` publishes, for every ICLR paper, the individual
reviewer ratings, the confidences, the per-review word counts and the final
status — everything except the review text. That is exactly the half of the
pipeline that cannot be checked by unit tests on invented fixtures.

So this script converts paperlists records into the schema that
`scripts/fetch_openreview.py` emits, substitutes placeholder text of the
**real** length (from `wc_review`), and runs the real Task 3 functions over
it. What comes out is a genuine projection of the pair yield, the per-year
accept thresholds, and the length confound — computed from real ICLR
decisions rather than assumed.

What this DOES validate:
  * the per-year accept-threshold estimator, on real rating scales
    (2020 is {1,3,6,8}, 2021 is 1-10, 2022+ is {1,3,5,6,8,10} — a fixed cut of
    5 would mean a different thing each year);
  * the straddle rule, the chosen/rejected assignment, and the majority logic,
    against real accept/reject outcomes;
  * the projected number of pairs, i.e. whether kill criterion 1 is cleared;
  * whether chosen reviews are systematically longer than rejected ones, which
    is the length confound H2 has to survive.

What it does NOT validate: anything about review text — the parsing of
OpenReview's per-year prose fields, and therefore the actual training data.
Only a real fetch does that.

Usage
-----
    git clone --depth 1 https://github.com/papercopilot/paperlists /tmp/paperlists
    python scripts/validate_on_paperlists.py --paperlists /tmp/paperlists/iclr

ICLR 2025 and 2026 are git-LFS pointers in that repo; `git lfs pull` retrieves
them. 2026 is excluded from pair construction by design regardless.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_pairs", _HERE / "build_pairs.py")
bp = importlib.util.module_from_spec(_spec)
sys.modules["build_pairs"] = bp
_spec.loader.exec_module(bp)

# paperlists names the score column differently in the years ICLR did.
RATING_KEY = {2020: "rating", 2021: "rating", 2022: "recommendation",
              2023: "recommendation", 2024: "rating", 2025: "rating"}


def _split_scores(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for chunk in str(raw or "").split(";"):
        chunk = chunk.strip()
        try:
            out.append(float(chunk))
        except ValueError:
            out.append(None)
    return out


def to_fetch_schema(paper: dict, year: int) -> dict:
    """Convert one paperlists row into a Task 2 output record.

    Review text is a placeholder padded to the real word count from
    `wc_review`, so length statistics are real even though the prose is not.
    Each placeholder is unique, so the identical-text guard does not fire
    spuriously.
    """
    ratings = _split_scores(paper.get(RATING_KEY.get(year, "rating")))
    confidences = _split_scores(paper.get("confidence"))
    word_counts = _split_scores(paper.get("wc_review"))
    status = (paper.get("status") or "").strip()
    paper_id = paper.get("id")

    reviews = []
    for i, rating in enumerate(ratings):
        if rating is None:
            continue
        n_words = word_counts[i] if i < len(word_counts) and word_counts[i] else 200
        filler = " ".join(f"w{j}" for j in range(max(1, int(n_words)) - 1))
        reviews.append({
            "review_id": f"{paper_id}#r{i}",
            "rating": rating,
            "rating_field": RATING_KEY.get(year, "rating"),
            "confidence": confidences[i] if i < len(confidences) else None,
            "text": f"[placeholder:{paper_id}#r{i}] {filler}".strip(),
            "text_fields": {},
        })

    rated = [r["rating"] for r in reviews]
    return {
        "paper_id": paper_id,
        "year": year,
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "primary_area": (paper.get("primary_area") or "").strip() or None,
        "venue": status,
        "decision": status or None,
        "decision_source": "paperlists_status",
        "accepted": bp_classify(status),
        "withdrawn": "withdraw" in status.lower(),
        "desk_rejected": "desk" in status.lower(),
        "reviews": reviews,
        "n_reviews": len(reviews),
        "n_ratings": len(rated),
        "rating_spread": (max(rated) - min(rated)) if len(rated) >= 2 else None,
        "meta_review": None,
    }


def bp_classify(status: str) -> bool | None:
    """Reuse the fetcher's own acceptance rule, so both stay in step."""
    fetch_spec = importlib.util.spec_from_file_location(
        "fetch_openreview", _HERE / "fetch_openreview.py"
    )
    global _FETCH
    try:
        _FETCH
    except NameError:
        _FETCH = importlib.util.module_from_spec(fetch_spec)
        fetch_spec.loader.exec_module(_FETCH)
    return _FETCH.classify_acceptance(status, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paperlists", required=True,
                   help="path to the paperlists 'iclr' directory")
    p.add_argument("--years", default="2020-2025")
    p.add_argument("--min-spread", type=float, default=1.0)
    p.add_argument("--out", default=str(bp.REPO_ROOT / "data" / "YIELD_PROJECTION.md"))
    args = p.parse_args()

    src = pathlib.Path(args.paperlists)
    years = bp.parse_years(args.years)
    rows, per_year = [], {}

    for year in years:
        path = src / f"iclr{year}.json"
        if not path.exists():
            print(f"  ! {path} missing; skipping", file=sys.stderr)
            continue
        if path.stat().st_size < 1000:
            print(f"  ! {path} looks like a git-LFS pointer ({path.stat().st_size} B); "
                  f"run `git lfs pull` to fetch it. Skipping.", file=sys.stderr)
            continue

        papers = json.loads(path.read_text(encoding="utf-8"))
        records = [to_fetch_schema(paper, year) for paper in papers]
        est = bp.estimate_threshold(records)
        threshold = est["threshold"]

        print(f"\n=== ICLR {year} ===")
        print(f"  {len(records):,} papers")
        if threshold is None:
            print(f"  ! no threshold ({est.get('note')})", file=sys.stderr)
            continue
        print(f"  accept threshold {threshold:.2f}  "
              f"(Youden J {est['youden_j']:.3f}, sens {est['sensitivity']:.2f} / "
              f"spec {est['specificity']:.2f})")

        counter: collections.Counter = collections.Counter()
        pairs = []
        for record in records:
            pair, reason = bp.make_pair(record, threshold, args.min_spread)
            if pair is not None:
                pairs.append(pair)
            else:
                counter[reason] += 1
        rows.extend(pairs)
        per_year[year] = {"est": est, "n_papers": len(records), "n_pairs": len(pairs),
                          "exclusions": counter}
        acc = sum(1 for x in pairs if x["accepted"])
        against = sum(1 for x in pairs if x["majority_agreed_with_decision"] is False)
        print(f"  {len(pairs):,} pairs  "
              f"({100.0 * len(pairs) / len(records):.1f}% of papers)")
        print(f"    accepted {acc:,} / rejected {len(pairs) - acc:,}")
        print(f"    against the reviewer majority: {against:,} "
              f"({100.0 * against / max(1, len(pairs)):.1f}%)")
        print(f"    top exclusions: "
              f"{', '.join(f'{k}={v}' for k, v in counter.most_common(3))}")

    if not rows:
        print("\nNo pairs projected.", file=sys.stderr)
        return 1

    # The length confound: if chosen reviews are reliably longer, a classifier
    # on length alone does well and H2 has real work to do.
    longer = sum(1 for r in rows if r["chosen_len_tokens"] > r["rejected_len_tokens"])
    chosen_len = statistics.fmean(r["chosen_len_tokens"] for r in rows)
    rejected_len = statistics.fmean(r["rejected_len_tokens"] for r in rows)
    n = len(rows)
    n_acc = sum(1 for r in rows if r["accepted"])
    against = sum(1 for r in rows if r["majority_agreed_with_decision"] is False)

    print("\n" + "=" * 70)
    print(f"PROJECTED TOTAL: {n:,} pairs across {len(per_year)} years")
    print(f"  accepted {n_acc:,} ({100.0 * n_acc / n:.1f}%) / "
          f"rejected {n - n_acc:,} ({100.0 * (n - n_acc) / n:.1f}%)")
    print(f"  against the reviewer majority: {against:,} ({100.0 * against / n:.1f}%)")
    print(f"  mean length: chosen {chosen_len:.0f} words vs rejected "
          f"{rejected_len:.0f} words")
    print(f"  chosen is the longer review in {100.0 * longer / n:.1f}% of pairs "
          f"(50% = no length confound; this is the always-longer baseline)")
    print(f"\nKill criterion 1: need >= 3,000 pairs. Projected {n:,}. "
          f"{'CLEARED' if n >= 3000 else 'NOT CLEARED'}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Projected pair yield (validation run, no review text)",
        "",
        "Generated by `scripts/validate_on_paperlists.py`. Do not edit by hand.",
        "",
        "Computed from `papercopilot/paperlists`, which publishes real per-reviewer",
        "ratings, confidences, word counts and final statuses for every ICLR paper,",
        "but **no review text**. The Task 3 construction logic is run unchanged over",
        "those real ratings and decisions, with placeholder text padded to the real",
        "word count. So the counts, thresholds and length statistics below are real;",
        "only the prose is not.",
        "",
        "This exists because OpenReview was unreachable when the pipeline was written.",
        "It validates everything except text parsing. Replace it with a real fetch",
        "before quoting any of it in the paper.",
        "",
        "## Yield by year",
        "",
        "| Year | Papers | Pairs | Yield | Accept threshold | Youden J | Sens | Spec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for year in sorted(per_year):
        d = per_year[year]
        e = d["est"]
        lines.append(
            f"| {year} | {d['n_papers']:,} | {d['n_pairs']:,} | "
            f"{100.0 * d['n_pairs'] / d['n_papers']:.1f}% | {e['threshold']:.2f} | "
            f"{e['youden_j']:.3f} | {e['sensitivity']:.2f} | {e['specificity']:.2f} |"
        )
    lines += [
        f"| **Total** | | **{n:,}** | | | | | |",
        "",
        "The threshold is estimated per year rather than fixed, because the rating",
        "scale itself changes: ICLR 2020 rates on {1,3,6,8}, 2021 on 1-10, and 2022",
        "onwards on {1,3,5,6,8,10}. One hardcoded cut would mean a different thing",
        "each year.",
        "",
        "## Balance and the honest-test subset",
        "",
        f"- Accepted: {n_acc:,} ({100.0 * n_acc / n:.1f}%) — "
        f"rejected: {n - n_acc:,} ({100.0 * (n - n_acc) / n:.1f}%)",
        f"- Decision went against the reviewer majority: {against:,} "
        f"({100.0 * against / n:.1f}%)",
        "",
        "The evaluation split must be balanced 50/50 (HANDOFF section 6); these are",
        "the raw construction counts, not an evaluation set.",
        "",
        "## The length confound",
        "",
        f"- Mean chosen length: **{chosen_len:.0f} words**",
        f"- Mean rejected length: **{rejected_len:.0f} words**",
        f"- Chosen is the longer review in **{100.0 * longer / n:.1f}%** of pairs",
        "",
        "That last number is the accuracy of a classifier that always picks the",
        "longer review. It belongs in the results table beside the always-negative",
        "baseline: if the tuned model does not clear it comfortably, the result is",
        "length, not judgement.",
        "",
        "## Exclusions",
        "",
        "| Year | " + " | ".join(bp.EXCLUSION_REASONS) + " |",
        "|---" * (len(bp.EXCLUSION_REASONS) + 1) + "|",
    ]
    for year in sorted(per_year):
        counter = per_year[year]["exclusions"]
        lines.append(f"| {year} | " + " | ".join(
            str(counter.get(r, 0)) for r in bp.EXCLUSION_REASONS) + " |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
