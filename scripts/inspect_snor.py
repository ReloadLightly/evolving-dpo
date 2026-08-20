#!/usr/bin/env python3
"""Report the schema of the SNOR OpenReview dump without loading it into RAM.

SNOR v1 (Structured, Normalized OpenReview), Mark Neumann, CC BY 4.0,
DOI 10.5281/zenodo.15866613 — ICLR 2017-2025 and NeurIPS 2021-2025, linked to
Semantic Scholar.

Run this on the machine holding the download and send back `snor_schema.json`.
It is a few KB, contains no bulk review text (string values are truncated to
200 characters), and answers the questions that decide whether SNOR can
replace the OpenReview fetch:

  * do comment records carry full review TEXT, or only structured metadata?
  * is there a per-review rating, and on what scale, per venue and year?
  * is there a paper-level decision, and a primary area?
  * can a comment be joined back to its paper?

Usage
-----
    python scripts/inspect_snor.py ~/Downloads
    python scripts/inspect_snor.py ~/Downloads --sample 20000

Streams line by line, so the 1.3 GB file costs memory proportional to one
record, not to the file.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

FILES = ("normalized_papers.jsonl", "normalized_comments.jsonl", "failed_matches.jsonl")

# Fields whose presence decides whether SNOR can stand in for a real fetch.
RATING_HINTS = ("rating", "recommendation", "score", "overall", "soundness",
                "presentation", "contribution", "confidence")
TEXT_HINTS = ("text", "review", "summary", "strength", "weakness", "question",
              "comment", "body", "content", "main_review")
LINK_HINTS = ("id", "forum", "paper", "submission", "parent", "reply")


def truncate(value, limit: int = 200):
    """Shrink a value for reporting; never emit bulk review text."""
    if isinstance(value, str):
        return value[:limit] + ("…" if len(value) > limit else "")
    if isinstance(value, list):
        return [truncate(v, 80) for v in value[:3]] + (["…"] if len(value) > 3 else [])
    if isinstance(value, dict):
        return {k: truncate(v, 80) for k, v in list(value.items())[:8]}
    return value


def describe(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, list):
        inner = describe(value[0]) if value else "empty"
        return f"list[{inner}]"
    if isinstance(value, dict):
        return f"dict{{{','.join(list(value)[:6])}}}"
    return type(value).__name__


def inspect(path: pathlib.Path, sample: int) -> dict:
    field_counts: collections.Counter = collections.Counter()
    field_types: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    venues: collections.Counter = collections.Counter()
    years: collections.Counter = collections.Counter()
    text_lengths: dict[str, list[int]] = collections.defaultdict(list)
    rating_values: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    first_record = None
    n = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            n += 1
            if first_record is None:
                first_record = {k: truncate(v) for k, v in rec.items()}
            for key, value in rec.items():
                field_counts[key] += 1
                field_types[key][describe(value)] += 1
                low = key.lower()
                if isinstance(value, str) and any(h in low for h in TEXT_HINTS):
                    text_lengths[key].append(len(value))
                if any(h in low for h in RATING_HINTS) and isinstance(value, (int, float, str)):
                    rating_values[key][str(value)[:40]] += 1
                if low in ("venue", "venueid", "conference"):
                    venues[str(value)[:60]] += 1
                if low in ("year", "venue_year"):
                    years[str(value)[:10]] += 1
            if n >= sample:
                break

    def summarize_lengths(values: list[int]) -> dict:
        values = sorted(values)
        if not values:
            return {}
        return {"n": len(values), "min": values[0], "median": values[len(values) // 2],
                "max": values[-1], "mean": round(sum(values) / len(values))}

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "records_sampled": n,
        "fields": {k: {"present_in": v, "types": dict(field_types[k].most_common(4))}
                   for k, v in field_counts.most_common()},
        "candidate_text_fields": {k: summarize_lengths(v)
                                  for k, v in sorted(text_lengths.items())},
        "candidate_rating_fields": {k: dict(v.most_common(12))
                                    for k, v in sorted(rating_values.items())},
        "venues": dict(venues.most_common(30)),
        "years": dict(years.most_common(20)),
        "join_key_candidates": sorted(
            k for k in field_counts if any(h in k.lower() for h in LINK_HINTS)
        ),
        "first_record_truncated": first_record,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="folder holding the three SNOR .jsonl files")
    p.add_argument("--sample", type=int, default=20000,
                   help="records to read per file (default 20000)")
    p.add_argument("--out", default="snor_schema.json")
    args = p.parse_args()

    directory = pathlib.Path(args.directory).expanduser()
    report = {"source": "SNOR v1, DOI 10.5281/zenodo.15866613, CC BY 4.0",
              "sample_per_file": args.sample, "files": {}}

    for name in FILES:
        path = directory / name
        if not path.exists():
            print(f"  ! {path} not found; skipping", file=sys.stderr)
            continue
        print(f"reading {path} ({path.stat().st_size / 1e9:.2f} GB) …")
        info = inspect(path, args.sample)
        report["files"][name] = info
        print(f"  {info['records_sampled']:,} records sampled, "
              f"{len(info['fields'])} distinct fields")
        if info["candidate_text_fields"]:
            for field, stats in info["candidate_text_fields"].items():
                if stats.get("median", 0) > 200:
                    print(f"    text-bearing: {field} "
                          f"(median {stats['median']} chars, max {stats['max']})")
        if info["candidate_rating_fields"]:
            print(f"    rating-like: {', '.join(list(info['candidate_rating_fields'])[:6])}")

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB) — send this back.")
    print("It holds field names, types and 200-char samples: no bulk review text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
