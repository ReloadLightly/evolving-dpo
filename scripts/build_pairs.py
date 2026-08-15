#!/usr/bin/env python3
"""Task 3: turn fetched ICLR reviews into institutionally-arbitrated pairs.

Reads data/raw/iclr{year}.jsonl (Task 2 output) and writes:

    data/pairs.jsonl        the pairs, with review text     (never committed)
    data/pairs_index.csv    identifiers and numbers only    (committed)
    data/DATASET_CARD.md    counts, thresholds, exclusions  (committed)

The index is the artifact we actually release. HANDOFF section 8 commits us to
publishing pair *identifiers* and construction code rather than a bulk
redistribution of CC BY 4.0 review text, so anyone can rebuild pairs.jsonl
from OpenReview with scripts/fetch_openreview.py and this script.

Usage
-----
    python scripts/build_pairs.py                     # all years found
    python scripts/build_pairs.py --years 2020-2023   # the training split
    python scripts/build_pairs.py --min-spread 3      # tighter disagreement

The construction rule
---------------------
For each submission whose reviewers disagreed *across the venue's accept
threshold*:

    chosen   = the most extreme review on the side the institution took
    rejected = the most extreme review on the losing side

Both are reviews of the same paper, from the same cycle, written to the same
instructions — style-matched by construction — and the arbiter is the venue's
own decision rather than the author or an LLM judge.

The threshold is estimated per year, never hardcoded. ICLR 2020 rates on
{1,3,6,8} and later years on 1-10, so a fixed cut of 5 would mean something
different every year. `estimate_threshold` picks the rating cut that best
separates that year's accepted papers from its rejected ones (Youden's J,
which is insensitive to the ~70% rejection rate), and the chosen cut and its
separation quality go in the dataset card.

Traps this script is built around (HANDOFF section 6)
-----------------------------------------------------
* The 70% trap. ~70% of submissions are rejected, and reviews recommending
  rejection read differently. This script does not balance — Task 5's
  evaluation split does — but it records `accepted` on every row so the split
  can, and the card reports the balance so nobody forgets.
* Agreement is not correctness. `majority_agreed_with_decision` marks the rows
  where the decision went against the numerical majority; that subset is the
  honest test and the card counts it.
* Ties are broken by review id, never by length. Breaking ties by longer text
  would quietly build the length confound into the labels that H2 exists to
  rule out.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = REPO_ROOT / "data"

# ICLR 2026 is excluded by design: after a security incident the organizers
# reset all scores to a pre-rebuttal state mid-discussion, so its record is not
# a clean institutional trace (HANDOFF section 5, Task 3).
EXCLUDED_YEARS = frozenset({2026})

EXCLUSION_REASONS = (
    "excluded_year",
    "withdrawn_or_desk_rejected",
    "no_decision",
    "too_few_rated_reviews",
    "no_straddle",
    "spread_below_min",
    "missing_text",
    "identical_text",
    "no_prompt",
)


# --------------------------------------------------------------------------
# Threshold estimation
# --------------------------------------------------------------------------

def mean_rating(record: dict) -> float | None:
    ratings = [r["rating"] for r in record.get("reviews", []) if r.get("rating") is not None]
    return statistics.fmean(ratings) if ratings else None


def estimate_threshold(records: list[dict]) -> dict[str, Any]:
    """Find the rating cut that best separates accepted from rejected papers.

    Scores each candidate cut by Youden's J (sensitivity + specificity - 1)
    rather than accuracy: with ~70% of papers rejected, plain accuracy is
    maximized by a degenerate cut that rejects nearly everything.

    Returns the cut and the diagnostics needed to judge whether it is
    trustworthy, so a year where ratings barely predict the outcome shows up
    as a low J rather than as a confident-looking number.
    """
    labelled = [
        (mean_rating(r), r["accepted"])
        for r in records
        if r.get("accepted") is not None and mean_rating(r) is not None
    ]
    if len(labelled) < 20:
        return {"threshold": None, "youden_j": None, "n_labelled": len(labelled),
                "sensitivity": None, "specificity": None, "note": "too few labelled papers"}

    values = sorted({m for m, _ in labelled})
    # Candidate cuts: every observed mean, plus the midpoints between them.
    candidates = list(values)
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates = sorted(set(candidates))

    n_pos = sum(1 for _, acc in labelled if acc)
    n_neg = len(labelled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"threshold": None, "youden_j": None, "n_labelled": len(labelled),
                "sensitivity": None, "specificity": None,
                "note": "only one outcome class present"}

    best = None
    for cut in candidates:
        tp = sum(1 for m, acc in labelled if acc and m >= cut)
        tn = sum(1 for m, acc in labelled if not acc and m < cut)
        sens, spec = tp / n_pos, tn / n_neg
        j = sens + spec - 1
        # Tie-break toward the lower cut for reproducibility.
        if best is None or j > best["youden_j"]:
            best = {"threshold": cut, "youden_j": j,
                    "sensitivity": sens, "specificity": spec}
    best.update({"n_labelled": len(labelled), "n_accepted": n_pos, "n_rejected": n_neg,
                 "note": ""})
    return best


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------

def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    return len(text.split())


def build_prompt(record: dict) -> str | None:
    """The prompt both reviews answer: the paper itself."""
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    if not title and not abstract:
        return None
    return f"{title}\n\n{abstract}".strip()


def _pick_extreme(reviews: list[dict], highest: bool) -> dict:
    """Most extreme review on one side; ties broken by id, never by length.

    Tie-breaking on text length would correlate the label with length and
    manufacture exactly the style confound H2 is designed to detect.
    """
    return sorted(
        reviews,
        key=lambda r: (-r["rating"] if highest else r["rating"], str(r.get("review_id") or "")),
    )[0]


def make_pair(record: dict, threshold: float, min_spread: float = 1.0,
              tokenizer=None) -> tuple[dict | None, str | None]:
    """Build one pair, or explain why this paper yields none.

    Returns (pair, None) on success and (None, reason) otherwise, where reason
    is one of EXCLUSION_REASONS. Every drop is counted, so the dataset card can
    account for the gap between submissions and pairs.
    """
    year = record.get("year")
    if year in EXCLUDED_YEARS:
        return None, "excluded_year"
    if record.get("withdrawn") or record.get("desk_rejected"):
        return None, "withdrawn_or_desk_rejected"

    accepted = record.get("accepted")
    if accepted is None:
        return None, "no_decision"

    rated = [r for r in record.get("reviews", []) if r.get("rating") is not None]
    if len(rated) < 2:
        return None, "too_few_rated_reviews"

    # The institution's side, and the losing side, split at the venue's cut.
    accept_side = [r for r in rated if r["rating"] >= threshold]
    reject_side = [r for r in rated if r["rating"] < threshold]
    if not accept_side or not reject_side:
        return None, "no_straddle"

    if accepted:
        chosen, rejected = _pick_extreme(accept_side, True), _pick_extreme(reject_side, False)
    else:
        chosen, rejected = _pick_extreme(reject_side, False), _pick_extreme(accept_side, True)

    spread = abs(chosen["rating"] - rejected["rating"])
    if spread < min_spread:
        return None, "spread_below_min"

    chosen_text = (chosen.get("text") or "").strip()
    rejected_text = (rejected.get("text") or "").strip()
    if not chosen_text or not rejected_text:
        return None, "missing_text"
    if chosen_text == rejected_text:
        return None, "identical_text"

    prompt = build_prompt(record)
    if not prompt:
        return None, "no_prompt"

    n_decision_side = len(accept_side) if accepted else len(reject_side)
    n_other_side = len(reject_side) if accepted else len(accept_side)
    if n_decision_side > n_other_side:
        majority_agreed: bool | None = True
    elif n_decision_side < n_other_side:
        majority_agreed = False
    else:
        majority_agreed = None  # an even split has no majority to agree

    return {
        "paper_id": record.get("paper_id"),
        "year": year,
        "primary_area": record.get("primary_area"),
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_rating": chosen["rating"],
        "rejected_rating": rejected["rating"],
        "decision": record.get("decision"),
        "accepted": bool(accepted),
        "majority_agreed_with_decision": majority_agreed,
        "chosen_len_tokens": count_tokens(chosen_text, tokenizer),
        "rejected_len_tokens": count_tokens(rejected_text, tokenizer),
        # Provenance and bookkeeping beyond the required schema.
        "chosen_review_id": chosen.get("review_id"),
        "rejected_review_id": rejected.get("review_id"),
        "chosen_confidence": chosen.get("confidence"),
        "rejected_confidence": rejected.get("confidence"),
        "n_reviews": len(rated),
        "n_accept_side": len(accept_side),
        "n_reject_side": len(reject_side),
        # How many same-side reviews we passed over by taking the extremes.
        "n_discarded_alternatives": len(rated) - 2,
        "rating_spread": spread,
        "mean_rating": mean_rating(record),
        "year_threshold": threshold,
        "decision_source": record.get("decision_source"),
    }, None


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


INDEX_COLUMNS = (
    "paper_id", "year", "primary_area", "chosen_review_id", "rejected_review_id",
    "chosen_rating", "rejected_rating", "accepted", "majority_agreed_with_decision",
    "chosen_len_tokens", "rejected_len_tokens", "n_reviews", "rating_spread",
    "year_threshold",
)


def write_index(path: pathlib.Path, pairs: list[dict]) -> None:
    """The releasable artifact: identifiers and numbers, no review text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    tmp.replace(path)


# --------------------------------------------------------------------------
# Dataset card
# --------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


def build_card(pairs: list[dict], thresholds: dict[int, dict],
               exclusions: dict[int, collections.Counter],
               n_input: dict[int, int], min_spread: float, len_method: str) -> str:
    years = sorted({p["year"] for p in pairs})
    by_year = collections.Counter(p["year"] for p in pairs)
    by_area = collections.Counter(p["primary_area"] or "(not recorded)" for p in pairs)
    n = len(pairs)
    n_acc = sum(1 for p in pairs if p["accepted"])
    against_majority = [p for p in pairs if p["majority_agreed_with_decision"] is False]
    tied = [p for p in pairs if p["majority_agreed_with_decision"] is None]

    lines = [
        "# Dataset card — institutionally-arbitrated ICLR review pairs",
        "",
        "Generated by `scripts/build_pairs.py`. Do not edit by hand.",
        "",
        "## What a row is",
        "",
        "Two reviews of the **same** ICLR paper whose ratings fell on opposite",
        "sides of that year's accept threshold. `chosen` is the review on the",
        "side the venue's decision took; `rejected` is the review on the losing",
        "side. Both were written in the same cycle to the same instructions, so",
        "the pair is style-matched by construction and the arbiter is the",
        "institution — not the author (cf. RbtAct, arXiv:2603.09723) and not an",
        "LLM judge.",
        "",
        f"**{n:,} pairs** across {len(years)} years"
        f" ({years[0]}–{years[-1]})." if years else f"**{n:,} pairs**.",
        "",
        "## Counts by year",
        "",
        "| Year | Pairs | Submissions read | Yield | Accept threshold | Youden J | Accepted |",
        "|---|---|---|---|---|---|---|",
    ]
    for year in years:
        t = thresholds.get(year, {})
        thr = t.get("threshold")
        j = t.get("youden_j")
        acc = sum(1 for p in pairs if p["year"] == year and p["accepted"])
        lines.append(
            f"| {year} | {by_year[year]:,} | {n_input.get(year, 0):,} | "
            f"{_pct(by_year[year], n_input.get(year, 0))} | "
            f"{thr if thr is None else f'{thr:.2f}'} | "
            f"{j if j is None else f'{j:.3f}'} | "
            f"{_pct(acc, by_year[year])} |"
        )
    lines += [
        "",
        "The accept threshold is estimated per year, not assumed: it is the",
        "mean-rating cut maximizing Youden's J against that year's real",
        "accept/reject outcomes. ICLR 2020 rates on {1,3,6,8} and later years on",
        "1–10, so one fixed cut would mean a different thing each year. A low",
        "Youden J for some year means ratings separate outcomes poorly there —",
        "read that year's pairs with more suspicion, not less.",
        "",
        "## Counts by primary area",
        "",
        "`primary_area` is only recorded by OpenReview from 2024 onwards, which",
        "constrains the H3 area split to the later years unless areas are",
        "derived from keywords for the earlier ones.",
        "",
        "| Area | Pairs |",
        "|---|---|",
    ]
    for area, count in by_area.most_common(30):
        lines.append(f"| {area} | {count:,} |")
    if len(by_area) > 30:
        lines.append(f"| … and {len(by_area) - 30} more areas | |")

    lines += [
        "",
        "## Balance, and the 70% trap",
        "",
        f"- Accepted papers: {n_acc:,} ({_pct(n_acc, n)})",
        f"- Rejected papers: {n - n_acc:,} ({_pct(n - n_acc, n)})",
        "",
        "About 70% of ICLR submissions are rejected, and reviews recommending",
        "rejection read differently from reviews recommending acceptance. A",
        "model that always prefers the more negative review would therefore",
        "score around 70% and look like a result. **The evaluation split must be",
        "balanced 50/50 between accepted and rejected papers** so that strategy",
        "scores 50%, and the always-negative baseline belongs in every table.",
        "This file does not balance anything; it records `accepted` on every row",
        "so that Task 5 can.",
        "",
        "## Agreement is not correctness",
        "",
        f"- Decision went **against** the numerical majority: {len(against_majority):,} "
        f"({_pct(len(against_majority), n)})",
        f"- Reviewers split evenly, no majority: {len(tied):,} ({_pct(len(tied), n)})",
        "",
        "A review that matched the decision may have matched it by predicting the",
        "majority rather than by being right. Accuracy on the against-the-majority",
        "subset is the honest test, which is why every row carries",
        "`majority_agreed_with_decision` (`true`/`false`, or `null` for an even",
        "split, which has no majority to agree with).",
        "",
        "## Exclusions",
        "",
        "| Year | " + " | ".join(EXCLUSION_REASONS) + " |",
        "|---" * (len(EXCLUSION_REASONS) + 1) + "|",
    ]
    for year in sorted(exclusions):
        counter = exclusions[year]
        lines.append(
            f"| {year} | " + " | ".join(str(counter.get(r, 0)) for r in EXCLUSION_REASONS) + " |"
        )
    lines += [
        "",
        "What each reason means:",
        "",
        "- `excluded_year` — ICLR 2026, excluded by design: the organizers reset",
        "  all scores to a pre-rebuttal state mid-discussion after a security",
        "  incident, so its record is not a clean institutional trace.",
        "- `withdrawn_or_desk_rejected` — no institutional judgement of the",
        "  reviews' merits was ever made.",
        "- `no_decision` — no Decision note, meta-review recommendation, or venue",
        "  string to read the verdict from.",
        "- `too_few_rated_reviews` — fewer than two reviews carrying a rating.",
        "- `no_straddle` — reviewers agreed relative to the threshold, so there",
        "  is no disagreement for the institution to have arbitrated.",
        f"- `spread_below_min` — rating gap below the configured minimum ({min_spread}).",
        "- `missing_text` / `identical_text` / `no_prompt` — degenerate rows.",
        "",
        "## Construction details",
        "",
        "- Where several reviews sit on one side, the **most extreme** on each",
        "  side is taken; `n_discarded_alternatives` records how many were passed",
        "  over.",
        "- Ties in rating are broken by review id, **never by text length**.",
        "  Preferring the longer review would build the length confound directly",
        "  into the labels that H2 exists to rule out.",
        f"- `*_len_tokens` counted by: **{len_method}**.",
        "- `prompt` is the paper's title and abstract, so both reviews answer the",
        "  same question.",
        "",
        "## Licence and release",
        "",
        "Reviews and comments on OpenReview are released **CC BY 4.0**, metadata",
        "**CC0** (OpenReview Terms of Use, last updated 2024-09-24).",
        "",
        "We release **`data/pairs_index.csv` — identifiers, ratings and labels —",
        "together with the construction code**, not `data/pairs.jsonl`, which",
        "holds the review text. Anyone can rebuild the text locally with",
        "`scripts/fetch_openreview.py` followed by `scripts/build_pairs.py`. This",
        "keeps the licence chain intact and attribution with the reviewers.",
        "",
        "Reviewers wrote these texts for authors and area chairs, not as",
        "preference-optimization data. We commit to: no reviewer",
        "de-anonymization; no per-reviewer analysis; and an explicit statement",
        "that this artifact **must not be used in a deployed reviewing system**",
        "(cf. arXiv:2605.03202 on automating peer review without rigorous",
        "evaluation).",
        "",
        "## Known limitations",
        "",
        "- One venue, one field, English only. Nothing here generalizes to",
        "  \"science\".",
        "- Agreement with the outcome is not correctness (see above).",
        "- An estimated ~21% of recent ICLR reviews may be AI-generated. This is",
        "  unfixable for 2024–2025; the response is a 2020–2022 sensitivity",
        "  analysis, not a fix.",
        "- The threshold is estimated from paper-level mean ratings and then",
        "  applied to individual reviews. Reviewer score scales are known not to",
        "  be comparable across areas (arXiv:2607.27209), so a single per-year cut",
        "  is an approximation — the same approximation H3 is about.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_years(spec: str) -> list[int]:
    years: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(chunk))
    return sorted(set(years))


def discover_years(raw_dir: pathlib.Path) -> list[int]:
    years = []
    for path in sorted(raw_dir.glob("iclr*.jsonl")):
        stem = path.stem.replace("iclr", "")
        if stem.isdigit():
            years.append(int(stem))
    return years


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", default=str(DEFAULT_RAW), help="directory of iclr{year}.jsonl")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    p.add_argument("--years", default=None, help="e.g. 2020-2023 (default: all found)")
    p.add_argument("--min-spread", type=float, default=1.0,
                   help="minimum rating gap between the two sides (default 1.0)")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the per-year estimated accept threshold")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id for exact token counts "
                        "(default: whitespace word count, no download)")
    args = p.parse_args()

    raw_dir, out_dir = pathlib.Path(args.raw), pathlib.Path(args.out)
    years = parse_years(args.years) if args.years else discover_years(raw_dir)
    if not years:
        print(f"No iclr*.jsonl found in {raw_dir}. Run scripts/fetch_openreview.py "
              f"first.", file=sys.stderr)
        return 1

    tokenizer, len_method = None, "whitespace word count"
    if args.tokenizer:
        from transformers import AutoTokenizer  # lazy: optional dependency
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        len_method = f"`{args.tokenizer}` tokenizer"

    all_pairs: list[dict] = []
    thresholds: dict[int, dict] = {}
    exclusions: dict[int, collections.Counter] = {}
    n_input: dict[int, int] = {}

    for year in years:
        path = raw_dir / f"iclr{year}.jsonl"
        if not path.exists():
            print(f"  ! {path} missing; skipping", file=sys.stderr)
            continue
        records = read_jsonl(path)
        n_input[year] = len(records)

        est = estimate_threshold(records)
        if args.threshold is not None:
            est = {**est, "threshold": args.threshold, "note": "overridden on the command line"}
        thresholds[year] = est
        threshold = est["threshold"]

        print(f"\n=== ICLR {year} ===")
        print(f"  {len(records):,} submissions read")
        if threshold is None:
            print(f"  ! no usable accept threshold ({est.get('note')}); skipping year. "
                  f"Pass --threshold to force one.", file=sys.stderr)
            exclusions[year] = collections.Counter({"no_decision": len(records)})
            continue
        print(f"  accept threshold {threshold:.2f} "
              f"(Youden J {est['youden_j']:.3f}, "
              f"sens {est['sensitivity']:.2f} / spec {est['specificity']:.2f}, "
              f"n={est['n_labelled']:,})")
        if est["youden_j"] is not None and est["youden_j"] < 0.3:
            print(f"  ! weak separation for {year}: ratings predict the outcome "
                  f"poorly, so the threshold is shaky", file=sys.stderr)

        counter: collections.Counter = collections.Counter()
        for record in records:
            pair, reason = make_pair(record, threshold, args.min_spread, tokenizer)
            if pair is not None:
                all_pairs.append(pair)
            else:
                counter[reason] += 1
        exclusions[year] = counter
        made = sum(1 for p in all_pairs if p["year"] == year)
        print(f"  {made:,} pairs "
              f"({_pct(made, len(records))} of submissions); "
              f"top exclusions: "
              f"{', '.join(f'{k}={v}' for k, v in counter.most_common(3))}")

    if not all_pairs:
        print("\nNo pairs built.", file=sys.stderr)
        return 1

    all_pairs.sort(key=lambda p: (p["year"], str(p["paper_id"])))

    pairs_path = out_dir / "pairs.jsonl"
    index_path = out_dir / "pairs_index.csv"
    card_path = out_dir / "DATASET_CARD.md"
    write_jsonl(pairs_path, all_pairs)
    write_index(index_path, all_pairs)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        build_card(all_pairs, thresholds, exclusions, n_input, args.min_spread, len_method),
        encoding="utf-8",
    )

    n = len(all_pairs)
    n_acc = sum(1 for p in all_pairs if p["accepted"])
    against = sum(1 for p in all_pairs if p["majority_agreed_with_decision"] is False)
    print("\n" + "=" * 70)
    print(f"{n:,} pairs -> {pairs_path}")
    print(f"          -> {index_path}  (identifiers only; this is what we release)")
    print(f"          -> {card_path}")
    print(f"  accepted {n_acc:,} ({_pct(n_acc, n)}) / rejected {n - n_acc:,} "
          f"({_pct(n - n_acc, n)})")
    print(f"  decision went against the reviewer majority in {against:,} pairs "
          f"({_pct(against, n)}) — the honest test subset")
    print(f"\nKill criterion 1 (HANDOFF section 8): need >= 3,000 pairs. "
          f"Have {n:,}. {'CLEARED' if n >= 3000 else 'NOT CLEARED — STOP'}")
    if n < 3000:
        print("  The preregistration says to abandon below 3,000 pairs rather "
              "than proceed underpowered.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
