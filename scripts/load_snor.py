#!/usr/bin/env python3
"""Convert the SNOR OpenReview dump into the schema Task 3 already consumes.

SNOR v1 (Structured, Normalized OpenReview), Mark Neumann, CC BY 4.0,
DOI 10.5281/zenodo.15866613 — ICLR 2017-2025, NeurIPS 2021-2025, linked to
Semantic Scholar.

SNOR is already normalized across the API v1/v2 split and the per-year field
drift that scripts/fetch_openreview.py exists to absorb, so it can stand in
for the OpenReview fetch: this script writes the same
`data/raw/iclr{year}.jsonl` records, and `scripts/build_pairs.py` then runs
unchanged.

    python scripts/load_snor.py ~/Downloads
    python scripts/build_pairs.py --years 2020-2025

The selection effect you must not ignore
----------------------------------------
SNOR links submissions to Semantic Scholar, and `failed_matches.jsonl` holds
the ones that failed. Those failures are overwhelmingly *rejected* and
*withdrawn* papers — rejected work often never gets published, so it has no
Semantic Scholar record. Since ~71% of our pairs come from rejected papers,
dropping the unmatched ones biases the dataset toward rejects that were
eventually published elsewhere.

This script therefore:
  * reports the accept rate per year for matched vs unmatched papers, so the
    size of the bias is measured rather than assumed;
  * can fold `failed_matches.jsonl` back in with --include-failed, recovering
    those papers from the raw OpenReview note SNOR preserved;
  * counts comments whose paper is absent from both files (orphans), which is
    review text we would otherwise silently lose.

Other differences from a real OpenReview fetch, all recorded in the output:
  * SNOR has no `primary_area`. The H3 area split needs another basis --
    `keywords` (kept on every record) or clustering the Specter embeddings
    SNOR ships. Do not assume the field is merely missing from some years.
  * NeurIPS publishes reviews for accepted papers only, so its pairs would be
    ~100% accepted and would invert the class imbalance rather than fix it.
    ICLR only, unless --include-neurips is passed deliberately.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the fetcher's text assembly and score parsing, so SNOR-sourced records
# and OpenReview-sourced records are built by exactly the same rules.
fo = _load("fetch_openreview")

PAPERS = "normalized_papers.jsonl"
COMMENTS = "normalized_comments.jsonl"
FAILED = "failed_matches.jsonl"


def stream_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def split_conf_id(conf_id: str) -> tuple[str, int | None]:
    """'iclr2017' -> ('iclr', 2017)."""
    conf_id = (conf_id or "").strip().lower()
    digits = "".join(c for c in conf_id if c.isdigit())
    name = "".join(c for c in conf_id if not c.isdigit())
    return name, int(digits) if len(digits) == 4 else None


def review_from_comment(comment: dict) -> dict | None:
    """One SNOR comment -> one review record, or None if it carries no text."""
    content = fo.normalize_content(comment.get("content") or {})
    text, text_fields = fo.assemble_review_text(content)
    if not text:
        return None

    rating = comment.get("numeric_rating")
    if rating is None:
        rating = fo.parse_score(comment.get("rating"))
    confidence = comment.get("numeric_confidence")
    if confidence is None:
        confidence = fo.parse_score(comment.get("confidence"))

    return {
        "review_id": comment.get("comment_id"),
        "invitation": None,
        "rating": float(rating) if rating is not None else None,
        "rating_field": "numeric_rating",
        "rating_raw": comment.get("rating"),
        "confidence": float(confidence) if confidence is not None else None,
        "aux_scores": {},
        "text": text,
        "text_fields": text_fields,
        "n_chars": len(text),
        "tcdate": None,
        "tmdate": None,
    }


def paper_record(paper: dict, source: str) -> dict | None:
    """One SNOR paper -> one Task 2 style submission record (reviews added later)."""
    if source == "matched":
        conf_id = paper.get("conf_id") or ""
        decision = paper.get("raw_decision") or paper.get("normalized_decision")
        title, abstract = paper.get("title"), paper.get("abstract")
    else:
        # failed_matches keeps the raw OpenReview note and a venueid instead.
        conf_id = paper.get("conference") or ""
        decision = paper.get("venueid")
        title, abstract = paper.get("title"), paper.get("abstract")

    name, year = split_conf_id(conf_id)
    if year is None:
        return None
    status = f"{decision or ''}".lower()

    return {
        "paper_id": paper.get("id"),
        "forum": paper.get("id"),
        "number": None,
        "year": year,
        "api_version": None,
        "snor_source": source,
        "conference": name,
        "title": title or "",
        "abstract": abstract or "",
        "keywords": paper.get("keywords") or [],
        # SNOR carries no primary_area; keywords and the Specter embedding are
        # the available bases for an area split.
        "primary_area": None,
        "venue": decision,
        "venueid": decision,
        "decision": decision,
        "decision_source": "snor_" + source,
        # SNOR's own boolean is authoritative; fall back to the venue string.
        "accepted": paper.get("accepted") if isinstance(paper.get("accepted"), bool)
        else fo.classify_acceptance(decision, None),
        "withdrawn": "withdraw" in status,
        "desk_rejected": "desk" in status,
        "citation_count": paper.get("citation_count"),
        "reviews": [],
        "meta_review": None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="folder holding the three SNOR .jsonl files")
    p.add_argument("--out", default=str(fo.DEFAULT_OUT), help="output directory")
    p.add_argument("--years", default="2017-2025")
    p.add_argument("--include-failed", action="store_true",
                   help="also load failed_matches.jsonl (recovers mostly-rejected "
                        "papers that SNOR could not link to Semantic Scholar)")
    p.add_argument("--include-neurips", action="store_true",
                   help="also emit NeurIPS (accepted-only reviews; read the "
                        "docstring before using this for the main study)")
    args = p.parse_args()

    src = pathlib.Path(args.directory).expanduser()
    out_dir = pathlib.Path(args.out)
    years = set(fo.parse_years(args.years))
    wanted = {"iclr"} | ({"neurips"} if args.include_neurips else set())

    papers_path, comments_path = src / PAPERS, src / COMMENTS
    for path in (papers_path, comments_path):
        if not path.exists():
            print(f"! {path} not found", file=sys.stderr)
            return 1

    # ---- papers -----------------------------------------------------------
    records: dict[str, dict] = {}
    accept_by_year: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    skipped_conf: collections.Counter = collections.Counter()

    print(f"reading {papers_path.name} …")
    for paper in stream_jsonl(papers_path):
        rec = paper_record(paper, "matched")
        if rec is None:
            continue
        if rec["conference"] not in wanted:
            skipped_conf[rec["conference"]] += 1
            continue
        if rec["year"] not in years:
            continue
        records[rec["paper_id"]] = rec
        accept_by_year[rec["year"]][("matched", bool(rec["accepted"]))] += 1
    print(f"  {len(records):,} papers kept")

    failed_ids: set[str] = set()
    failed_path = src / FAILED
    if failed_path.exists():
        print(f"reading {failed_path.name} …")
        n_failed_seen = 0
        for paper in stream_jsonl(failed_path):
            rec = paper_record(paper, "failed_match")
            if rec is None or rec["conference"] not in wanted or rec["year"] not in years:
                continue
            n_failed_seen += 1
            accept_by_year[rec["year"]][("failed", bool(rec["accepted"]))] += 1
            if args.include_failed and rec["paper_id"] not in records:
                records[rec["paper_id"]] = rec
                failed_ids.add(rec["paper_id"])
        print(f"  {n_failed_seen:,} unmatched papers seen"
              + (f", {len(failed_ids):,} folded in" if args.include_failed
                 else " (not loaded; pass --include-failed)"))

    # ---- comments ---------------------------------------------------------
    print(f"reading {comments_path.name} (this is the 1.3 GB one) …")
    n_comments = n_reviews = n_attached = n_orphan = n_no_text = 0
    content_keys: collections.Counter = collections.Counter()
    for comment in stream_jsonl(comments_path):
        n_comments += 1
        if not comment.get("is_review"):
            continue
        n_reviews += 1
        paper_id = comment.get("paper_id")
        rec = records.get(paper_id)
        if rec is None:
            n_orphan += 1
            continue
        review = review_from_comment(comment)
        if review is None:
            n_no_text += 1
            continue
        content_keys[tuple(sorted(review["text_fields"]))] += 1
        rec["reviews"].append(review)
        n_attached += 1

    print(f"  {n_comments:,} comments, {n_reviews:,} flagged is_review, "
          f"{n_attached:,} attached, {n_orphan:,} orphaned, {n_no_text:,} without text")

    # ---- finalize and write ----------------------------------------------
    by_year: dict[int, list[dict]] = collections.defaultdict(list)
    for rec in records.values():
        ratings = [r["rating"] for r in rec["reviews"] if r["rating"] is not None]
        rec["n_reviews"] = len(rec["reviews"])
        rec["n_ratings"] = len(ratings)
        rec["rating_spread"] = (max(ratings) - min(ratings)) if len(ratings) >= 2 else None
        by_year[rec["year"]].append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for year in sorted(by_year):
        rows = by_year[year]
        path = out_dir / f"iclr{year}.jsonl"
        fo.write_jsonl(path, rows)
        summary = fo.summarize(year, rows)
        summaries.append(summary)
        print(f"  {year}: {len(rows):,} papers -> {path}")

    # ---- the bias report --------------------------------------------------
    print("\n" + "=" * 74)
    print("SELECTION EFFECT: accept rate, Semantic-Scholar-matched vs unmatched")
    print("=" * 74)
    print(f"{'Year':<6}{'matched':>9}{'acc%':>7}{'unmatched':>11}{'acc%':>7}"
          f"{'  bias (pp)':>12}")
    print("-" * 74)
    bias_rows = []
    for year in sorted(accept_by_year):
        c = accept_by_year[year]
        m_yes, m_no = c[("matched", True)], c[("matched", False)]
        f_yes, f_no = c[("failed", True)], c[("failed", False)]
        m_tot, f_tot = m_yes + m_no, f_yes + f_no
        m_pct = 100.0 * m_yes / m_tot if m_tot else float("nan")
        f_pct = 100.0 * f_yes / f_tot if f_tot else float("nan")
        bias = m_pct - f_pct if (m_tot and f_tot) else float("nan")
        bias_rows.append({"year": year, "matched": m_tot, "matched_accept_pct": m_pct,
                          "unmatched": f_tot, "unmatched_accept_pct": f_pct,
                          "bias_pp": bias})
        print(f"{year:<6}{m_tot:>9}{m_pct:>7.1f}{f_tot:>11}{f_pct:>7.1f}{bias:>12.1f}")
    print("-" * 74)
    print("A large positive bias means the unmatched papers are far more often")
    print("rejections, so excluding them skews the dataset toward accepted work.")
    print("Rerun with --include-failed to fold them back in.")

    print("\nReview content field-sets seen (SNOR's per-year prose shapes):")
    for keys, count in content_keys.most_common(10):
        print(f"  {count:>7,}  {', '.join(keys) if keys else '(none)'}")

    if n_orphan:
        print(f"\n! {n_orphan:,} reviews belong to papers not loaded — that is review "
              f"text being dropped. Pass --include-failed and widen --years.")

    report = {
        "source": "SNOR v1, DOI 10.5281/zenodo.15866613, CC BY 4.0",
        "conferences": sorted(wanted),
        "include_failed": args.include_failed,
        "papers_loaded": len(records),
        "comments_total": n_comments,
        "reviews_flagged": n_reviews,
        "reviews_attached": n_attached,
        "reviews_orphaned": n_orphan,
        "reviews_without_text": n_no_text,
        "primary_area_available": False,
        "selection_effect": bias_rows,
        "content_field_sets": {", ".join(k) or "(none)": v
                               for k, v in content_keys.most_common(20)},
        "per_year": summaries,
        "skipped_conferences": dict(skipped_conf),
    }
    report_path = out_dir / "snor_load_report.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print("\nNext:  python scripts/build_pairs.py --years 2020-2025")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
