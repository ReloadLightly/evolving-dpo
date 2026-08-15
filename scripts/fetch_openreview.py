#!/usr/bin/env python3
"""Task 2: fetch ICLR submissions + review text from OpenReview.

Two stages, separately runnable, because the download is the expensive part
and the parse is the part you will iterate on:

    download   OpenReview API  ->  data/raw/cache/iclr{year}.json.gz   (verbatim)
    parse      that cache      ->  data/raw/iclr{year}.jsonl           (normalized)

Examples
--------
Check the API is reachable at all before committing to a long pull::

    python scripts/fetch_openreview.py --check

Small end-to-end smoke test (~1 min, one year, 25 papers)::

    python scripts/fetch_openreview.py --years 2023 --limit 25

The real pull (hours; resumable — already-cached years are skipped)::

    python scripts/fetch_openreview.py --years 2020-2025

Re-parse from cache after changing the extractor (no network)::

    python scripts/fetch_openreview.py --years 2020-2025 --reparse

Discover which content fields a year actually uses, before trusting the
extractor on it (this is the "detect, don't assume" step)::

    python scripts/fetch_openreview.py --years 2022 --census

Two API versions
----------------
ICLR <= 2023 lives on api.openreview.net (v1), ICLR >= 2024 on
api2.openreview.net (v2). They differ in three ways that matter here:

  * content shape: v1 ``{"rating": "8: accept"}``,
                   v2 ``{"rating": {"value": 8}}``;
  * invitations:   v1 ``note["invitation"]`` (str),
                   v2 ``note["invitations"]`` (list);
  * replies:       v1 ``details["directReplies"]``,
                   v2 ``details["replies"]``.

Everything below normalizes those away before any field is read.

Field names are NOT assumed. The rating field is ``recommendation`` in
2022/2023 and ``rating`` elsewhere; the prose is one ``review`` field in
2020/2021 but four or five separate fields from 2022 on. The extractor
detects what is present per review and records what it found, so a year with
an unexpected schema shows up as a warning rather than as silently empty text.

Note for anyone modifying the cache format: openreview-py's ``Note.to_json()``
drops the ``details`` key (verified in openreview-py 2.4.2 source), and the
replies — i.e. all the review text — live in ``details``. ``_note_to_cacheable``
below re-attaches them. Do not replace it with a bare ``to_json()``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Callable, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"

V1_BASEURL = "https://api.openreview.net"
V2_BASEURL = "https://api2.openreview.net"
V2_FIRST_YEAR = 2024  # ICLR 2024 onwards is API v2

# Yield measured from papercopilot/paperlists in Task 1 (HANDOFF section 3.1).
# The fetch is expected to land within a couple of percent of "with_ratings";
# a large shortfall means the pull is incomplete, not that the venue changed.
REFERENCE_YIELD = {
    2020: {"papers": 2594, "with_ratings": 2561, "spread_ge_3": 1225, "spread_ge_4": 603},
    2021: {"papers": 3014, "with_ratings": 2979, "spread_ge_3": 1001, "spread_ge_4": 379},
    2022: {"papers": 3422, "with_ratings": 3375, "spread_ge_3": 1572, "spread_ge_4": 512},
    2023: {"papers": 4955, "with_ratings": 4919, "spread_ge_3": 2314, "spread_ge_4": 818},
    2024: {"papers": 7407, "with_ratings": 7261, "spread_ge_3": 3474, "spread_ge_4": 1159},
    # 2025 was never counted (git-lfs blocked in the Task 1 sandbox); the
    # preregistration's pool figure is ~11,565 submissions.
    2025: {"papers": 11565, "with_ratings": None, "spread_ge_3": None, "spread_ge_4": None},
}

# Submission invitations, tried in order until one returns notes. ICLR used
# Blind_Submission through 2023 and Submission from 2024.
SUBMISSION_INVITATIONS = ("Blind_Submission", "Submission")

# The per-review score field, by preference. 2022/2023 call it "recommendation";
# every other year calls it "rating". Both are 1-10 on ICLR.
RATING_FIELDS = ("rating", "recommendation", "overall_score", "score")
CONFIDENCE_FIELDS = ("confidence",)

# Numeric side-scores worth keeping as metadata (they vary by year, and Task 3
# may want them), but which must never be concatenated into the review prose.
AUX_SCORE_FIELDS = (
    "soundness", "presentation", "contribution",           # 2024+
    "correctness", "technical_novelty_and_reproducibility",  # 2022/2023
    "empirical_novelty_and_significance",
)

# Never part of the review text: scores, checkboxes, admin, and the ethics
# free-text field (which we deliberately do not carry into training data).
NON_TEXT_FIELDS = frozenset(
    RATING_FIELDS + CONFIDENCE_FIELDS + AUX_SCORE_FIELDS + (
        "title", "paperhash", "code_of_conduct", "code_of_ethics",
        "ethics_flag", "ethics_review_area", "flag_for_ethics_review",
        "details_of_ethics_concerns", "reviewed_version", "venue", "venueid",
        "submission_number", "first_time_reviewer", "reviewer_certifications",
    )
)

# "8: Top 50% of accepted papers" -> 8 ; "3" -> 3 ; 8 -> 8
_SCORE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------
# Shape normalization: v1 vs v2
# --------------------------------------------------------------------------

def unwrap(value: Any) -> Any:
    """v2 wraps every content value as {"value": x}; v1 does not."""
    if isinstance(value, dict) and "value" in value and len(value) == 1:
        return value["value"]
    return value


def normalize_content(content: Any) -> dict[str, Any]:
    """Return a plain {field: value} dict for either API version."""
    if not isinstance(content, dict):
        return {}
    return {k: unwrap(v) for k, v in content.items()}


def note_invitations(note: dict) -> list[str]:
    """v1 stores one invitation string, v2 a list. Return a list either way."""
    if "invitations" in note and note["invitations"]:
        inv = note["invitations"]
        return list(inv) if isinstance(inv, (list, tuple)) else [str(inv)]
    inv = note.get("invitation")
    return [inv] if inv else []


def note_replies(note: dict) -> list[dict]:
    """v1 puts direct replies under details['directReplies'], v2 under 'replies'."""
    details = note.get("details") or {}
    replies = details.get("replies")
    if replies is None:
        replies = details.get("directReplies")
    return list(replies or [])


def _invitation_endswith(note: dict, *names: str) -> bool:
    """True if any of this note's invitations ends in /-/<name>.

    Matches the suffix rather than searching the whole string, so
    ``.../-/Official_Review`` matches but ``.../-/Official_Review_Revision``
    does not.
    """
    for inv in note_invitations(note):
        tail = inv.rsplit("/-/", 1)[-1] if "/-/" in inv else inv
        if tail in names:
            return True
    return False


def is_official_review(note: dict) -> bool:
    return _invitation_endswith(note, "Official_Review", "Review")


def is_decision(note: dict) -> bool:
    return _invitation_endswith(note, "Decision", "Acceptance_Decision")


def is_meta_review(note: dict) -> bool:
    return _invitation_endswith(note, "Meta_Review", "Meta_Review_Revision")


# --------------------------------------------------------------------------
# Field extraction
# --------------------------------------------------------------------------

def parse_score(raw: Any) -> float | None:
    """Pull the leading number out of a rating value of any shape."""
    raw = unwrap(raw)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        m = _SCORE_RE.match(raw)
        if m:
            return float(m.group(1))
    return None


def detect_field(content: dict, candidates: Iterable[str]) -> str | None:
    """First candidate field actually present with a parseable score."""
    for name in candidates:
        if name in content and parse_score(content[name]) is not None:
            return name
    return None


def humanize(field: str) -> str:
    return field.replace("_", " ").strip().capitalize()


def assemble_review_text(content: dict) -> tuple[str, dict[str, str]]:
    """Join the prose fields of one review into a single document.

    Returns (text, {field: text}). The per-field dict is kept so Task 3 can
    recombine sections differently — e.g. weaknesses only — without going
    back to the network.

    Field order follows the API's own schema order, which is stable per year,
    so the same review always serializes identically.
    """
    fields: dict[str, str] = {}
    for name, value in content.items():
        if name in NON_TEXT_FIELDS:
            continue
        value = unwrap(value)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) < 3:
            continue
        # A short "3: good" style string is a score that the year gave an
        # unfamiliar name; do not let it into the prose.
        if len(text) < 60 and _SCORE_RE.match(text) and ":" in text[:4]:
            continue
        fields[name] = text

    if len(fields) == 1:
        # 2020/2021: a single "review" field, no headings to add.
        return next(iter(fields.values())), fields
    parts = [f"## {humanize(name)}\n\n{text}" for name, text in fields.items()]
    return "\n\n".join(parts), fields


def parse_review(note: dict) -> dict | None:
    """Normalize one Official_Review note. None if it carries no usable text."""
    content = normalize_content(note.get("content"))
    rating_field = detect_field(content, RATING_FIELDS)
    conf_field = detect_field(content, CONFIDENCE_FIELDS)
    text, text_fields = assemble_review_text(content)
    if not text:
        return None

    aux = {}
    for name in AUX_SCORE_FIELDS:
        score = parse_score(content.get(name)) if name in content else None
        if score is not None:
            aux[name] = score

    return {
        "review_id": note.get("id"),
        "invitation": (note_invitations(note) or [None])[0],
        "rating": parse_score(content.get(rating_field)) if rating_field else None,
        "rating_field": rating_field,
        "rating_raw": unwrap(content.get(rating_field)) if rating_field else None,
        "confidence": parse_score(content.get(conf_field)) if conf_field else None,
        "aux_scores": aux,
        "text": text,
        "text_fields": text_fields,
        "n_chars": len(text),
        "tcdate": note.get("tcdate") or note.get("cdate"),
        "tmdate": note.get("tmdate") or note.get("mdate"),
    }


def _decision_string(content: dict) -> str | None:
    for key in ("decision", "recommendation", "final_decision"):
        value = unwrap(content.get(key))
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def classify_acceptance(decision: str | None, venue: str | None) -> bool | None:
    """True accepted / False rejected / None unknown.

    Checks 'reject' before 'accept' because "Reject (was Accept)"-style
    strings exist, and desk-rejection strings contain both words.
    """
    for text in (decision, venue):
        if not text:
            continue
        # Normalize separators: the outcome vocabulary is written both as
        # "Top-25%" and "notable top 25%" depending on year and source.
        low = re.sub(r"[-_/]+", " ", text.lower())
        if "withdraw" in low or "desk" in low:
            return None
        if "reject" in low or low.startswith("submitted to"):
            return False
        if "accept" in low or any(
            k in low for k in ("poster", "spotlight", "oral", "talk", "notable",
                               "top 5%", "top 25%")
        ):
            return True
    return None


def extract_decision(content: dict, replies: list[dict]) -> tuple[str | None, str | None]:
    """Find the institution's verdict, and say where it came from.

    Priority: an explicit Decision note (2021+), then the Meta_Review's
    recommendation (which is where ICLR 2020 records it), then the
    submission's own venue string (v2 encodes the outcome there, and it stays
    public even when the Decision note is not).
    """
    for note in replies:
        if is_decision(note):
            found = _decision_string(normalize_content(note.get("content")))
            if found:
                return found, "Decision"
    for note in replies:
        if is_meta_review(note):
            found = _decision_string(normalize_content(note.get("content")))
            if found:
                return found, "Meta_Review"
    venue = unwrap(content.get("venue"))
    if isinstance(venue, str) and venue.strip():
        return venue.strip(), "venue"
    return None, None


def parse_submission(note: dict, year: int) -> dict:
    """Normalize one submission and its reviews into an output record."""
    content = normalize_content(note.get("content"))
    replies = note_replies(note)

    reviews = []
    for reply in replies:
        if not is_official_review(reply):
            continue
        parsed = parse_review(reply)
        if parsed is not None:
            reviews.append(parsed)

    meta_review = None
    for reply in replies:
        if is_meta_review(reply):
            text, _ = assemble_review_text(normalize_content(reply.get("content")))
            if text:
                meta_review = text
            break

    decision, decision_source = extract_decision(content, replies)
    venue = unwrap(content.get("venue"))
    venueid = unwrap(content.get("venueid"))
    status_text = " ".join(str(x) for x in (decision, venue, venueid) if x).lower()

    ratings = [r["rating"] for r in reviews if r["rating"] is not None]
    authors = unwrap(content.get("authors"))

    return {
        "paper_id": note.get("id"),
        "forum": note.get("forum") or note.get("id"),
        "number": note.get("number"),
        "year": year,
        "api_version": 2 if year >= V2_FIRST_YEAR else 1,
        "title": unwrap(content.get("title")),
        "abstract": unwrap(content.get("abstract")),
        "keywords": unwrap(content.get("keywords")),
        # 2024+ ships primary_area; earlier years have no area field at all,
        # which is why the H3 area split is only defined on the later years.
        "primary_area": unwrap(content.get("primary_area")),
        "venue": venue,
        "venueid": venueid,
        "decision": decision,
        "decision_source": decision_source,
        "accepted": classify_acceptance(decision, venue),
        "withdrawn": "withdraw" in status_text,
        "desk_rejected": "desk" in status_text,
        "n_reviews": len(reviews),
        "n_ratings": len(ratings),
        "rating_spread": (max(ratings) - min(ratings)) if len(ratings) >= 2 else None,
        "n_authors": len(authors) if isinstance(authors, list) else None,
        "reviews": reviews,
        "meta_review": meta_review,
    }


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _retry(fn: Callable, *args, tries: int = 5, base_delay: float = 2.0,
           what: str = "request", **kwargs):
    """Exponential backoff (2s, 4s, 8s, 16s) on transient API failures.

    A 403 is not transient — it is either an egress policy denial or a
    permissions problem — so it is raised immediately rather than retried.
    """
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # openreview raises its own exception types
            text = str(exc)
            if "403" in text or "Forbidden" in text:
                raise
            if attempt == tries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  ! {what} failed ({exc.__class__.__name__}: {text[:120]}); "
                  f"retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)


def make_client(year: int):
    """Anonymous client for the right API version. Credentials optional."""
    import openreview  # lazy: the parse path must work without the package

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if year >= V2_FIRST_YEAR:
        return openreview.api.OpenReviewClient(
            baseurl=V2_BASEURL, username=username, password=password
        )
    return openreview.Client(baseurl=V1_BASEURL, username=username, password=password)


def _note_to_cacheable(note) -> dict:
    """Serialize a Note WITH its replies.

    openreview-py's to_json() omits `details` (and, in v2, `number`), and the
    replies live in `details`. Dropping them here would mean re-downloading
    everything to recover the review text.
    """
    body = dict(note.to_json())
    body.setdefault("number", getattr(note, "number", None))
    details = getattr(note, "details", None)
    if details:
        body["details"] = details
    return body


def download_year(year: int, limit: int | None = None, verbose: bool = True) -> list[dict]:
    """Fetch every submission of one ICLR year, replies included."""
    client = make_client(year)
    details = "replies" if year >= V2_FIRST_YEAR else "directReplies"

    notes = []
    used_invitation = None
    for name in SUBMISSION_INVITATIONS:
        invitation = f"ICLR.cc/{year}/Conference/-/{name}"
        if verbose:
            print(f"  trying invitation {invitation} (details={details})")
        try:
            notes = _retry(
                client.get_all_notes, invitation=invitation, details=details,
                what=f"get_all_notes {year}",
            )
        except Exception as exc:
            print(f"  ! {invitation} raised {exc.__class__.__name__}: "
                  f"{str(exc)[:160]}", file=sys.stderr)
            notes = []
        if notes:
            used_invitation = invitation
            break

    if not notes:
        raise RuntimeError(
            f"No submissions returned for ICLR {year}. Tried "
            f"{', '.join(SUBMISSION_INVITATIONS)}. If other years worked, this "
            f"year uses a different invitation name — list them with "
            f"client.get_all_invitations(regex=f'ICLR.cc/{year}/Conference/-/.*')."
        )

    if verbose:
        print(f"  {len(notes)} submissions from {used_invitation}")
    if limit:
        notes = notes[:limit]

    records = [_note_to_cacheable(n) for n in notes]

    # If replies did not arrive inline, fall back to one forum query per
    # paper. Correct but far slower, so say so loudly.
    missing = [r for r in records if not note_replies(r)]
    if missing and len(missing) > 0.5 * len(records):
        print(f"  ! inline replies missing for {len(missing)}/{len(records)} "
              f"submissions; falling back to per-forum queries (slow)",
              file=sys.stderr)
        for i, rec in enumerate(missing, 1):
            forum = rec.get("forum") or rec.get("id")
            try:
                replies = _retry(client.get_all_notes, forum=forum,
                                 what=f"forum {forum}")
            except Exception as exc:
                print(f"  ! forum {forum}: {exc}", file=sys.stderr)
                continue
            rec.setdefault("details", {})["replies"] = [
                _note_to_cacheable(n) for n in replies if n.id != rec.get("id")
            ]
            if verbose and i % 100 == 0:
                print(f"    {i}/{len(missing)} forums")
            time.sleep(0.05)  # be polite to a free public API

    return records


# --------------------------------------------------------------------------
# Cache and output I/O
# --------------------------------------------------------------------------

def cache_path(out_dir: pathlib.Path, year: int) -> pathlib.Path:
    return out_dir / "cache" / f"iclr{year}.json.gz"


def write_cache(path: pathlib.Path, records: list[dict]) -> None:
    """Write gzipped JSON atomically, so an interrupted run leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(records, fh)
    tmp.replace(path)


def read_cache(path: pathlib.Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def field_census(raw: list[dict]) -> dict[str, Any]:
    """Count which content fields the reviews of this year actually use.

    Run this before trusting the extractor on a year nobody has looked at:
    a rating field this script does not know about shows up here as a
    frequent field with no home in RATING_FIELDS.
    """
    review_fields: dict[str, int] = {}
    sub_fields: dict[str, int] = {}
    rating_fields: dict[str, int] = {}
    n_reviews = 0
    for note in raw:
        for key in normalize_content(note.get("content")):
            sub_fields[key] = sub_fields.get(key, 0) + 1
        for reply in note_replies(note):
            if not is_official_review(reply):
                continue
            n_reviews += 1
            content = normalize_content(reply.get("content"))
            for key in content:
                review_fields[key] = review_fields.get(key, 0) + 1
            detected = detect_field(content, RATING_FIELDS)
            rating_fields[str(detected)] = rating_fields.get(str(detected), 0) + 1
    return {
        "n_submissions": len(raw),
        "n_official_reviews": n_reviews,
        "submission_fields": dict(sorted(sub_fields.items(), key=lambda kv: -kv[1])),
        "review_fields": dict(sorted(review_fields.items(), key=lambda kv: -kv[1])),
        "detected_rating_field": rating_fields,
    }


def summarize(year: int, records: list[dict]) -> dict[str, Any]:
    with_ratings = [r for r in records if r["n_ratings"] >= 2]
    spreads = [r["rating_spread"] for r in with_ratings]
    live = [r for r in records if not r["withdrawn"] and not r["desk_rejected"]]
    return {
        "year": year,
        "papers": len(records),
        "with_ratings": len(with_ratings),
        "spread_ge_3": sum(1 for s in spreads if s >= 3),
        "spread_ge_4": sum(1 for s in spreads if s >= 4),
        "non_withdrawn": len(live),
        "with_decision": sum(1 for r in records if r["decision"]),
        "with_primary_area": sum(1 for r in records if r["primary_area"]),
        "accepted": sum(1 for r in records if r["accepted"] is True),
        "rejected": sum(1 for r in records if r["accepted"] is False),
        "total_reviews": sum(r["n_reviews"] for r in records),
        "reviews_with_text": sum(
            1 for r in records for rev in r["reviews"] if rev["text"]
        ),
    }


def print_report(summaries: list[dict]) -> None:
    """Print measured counts beside the Task 1 reference, with the delta."""
    print("\n" + "=" * 78)
    print("MEASURED vs REFERENCE (HANDOFF section 3.1, papercopilot)")
    print("=" * 78)
    head = f"{'Year':<6}{'papers':>8}{'ref':>8}{'delta':>8}   " \
           f"{'>=2 rat':>8}{'sp>=3':>7}{'ref':>7}{'sp>=4':>7}{'ref':>7}"
    print(head)
    print("-" * 78)
    for s in summaries:
        ref = REFERENCE_YIELD.get(s["year"], {})
        ref_papers = ref.get("papers")
        if ref_papers:
            delta = 100.0 * (s["papers"] - ref_papers) / ref_papers
            delta_s, ref_s = f"{delta:+.1f}%", str(ref_papers)
        else:
            delta_s, ref_s = "n/a", "n/a"
        print(f"{s['year']:<6}{s['papers']:>8}{ref_s:>8}{delta_s:>8}   "
              f"{s['with_ratings']:>8}{s['spread_ge_3']:>7}"
              f"{str(ref.get('spread_ge_3') or 'n/a'):>7}"
              f"{s['spread_ge_4']:>7}{str(ref.get('spread_ge_4') or 'n/a'):>7}")
    print("-" * 78)
    total3 = sum(s["spread_ge_3"] for s in summaries)
    total4 = sum(s["spread_ge_4"] for s in summaries)
    print(f"{'TOTAL':<6}{sum(s['papers'] for s in summaries):>8}"
          f"{'':>8}{'':>8}   {sum(s['with_ratings'] for s in summaries):>8}"
          f"{total3:>7}{'':>7}{total4:>7}")

    print("\nText and label coverage (both must be high or Task 3 will silently shrink):")
    for s in summaries:
        pct_text = 100.0 * s["reviews_with_text"] / max(1, s["total_reviews"])
        pct_dec = 100.0 * s["with_decision"] / max(1, s["papers"])
        pct_area = 100.0 * s["with_primary_area"] / max(1, s["papers"])
        print(f"  {s['year']}: {s['total_reviews']:>6} reviews, "
              f"{pct_text:5.1f}% with text | {pct_dec:5.1f}% papers with a decision "
              f"| {pct_area:5.1f}% with primary_area "
              f"| {s['accepted']} acc / {s['rejected']} rej")

    print(f"\nKill criterion 1 (HANDOFF section 8): need >= 3,000 pairs. "
          f"Papers with spread >= 3: {total3}. "
          f"{'CLEARED' if total3 >= 3000 else 'NOT CLEARED'}")
    print("Note: spread counts here are upper bounds on pair yield — Task 3 "
          "additionally requires the ratings to straddle the accept threshold.")


# --------------------------------------------------------------------------
# Connectivity preflight
# --------------------------------------------------------------------------

def check_connectivity(verbose: bool = True) -> bool:
    """Probe both API hosts and explain a failure precisely.

    Worth running first: the failure mode this project keeps hitting is an
    egress policy denial, which looks nothing like an OpenReview outage.
    """
    import urllib.error
    import urllib.request

    ok = True
    for label, url in (("v1", f"{V1_BASEURL}/notes?limit=1"),
                       ("v2", f"{V2_BASEURL}/notes?limit=1")):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                if verbose:
                    print(f"  {label} {url} -> HTTP {resp.status} OK")
        except urllib.error.HTTPError as exc:
            # The endpoint answered, which is all we need to know.
            if verbose:
                print(f"  {label} {url} -> HTTP {exc.code} (host reachable)")
            if exc.code in (403, 407):
                ok = False
                print(f"  ! {label}: {exc.code} — likely an egress proxy denial, "
                      f"not OpenReview.", file=sys.stderr)
        except Exception as exc:
            ok = False
            if verbose:
                print(f"  {label} {url} -> UNREACHABLE ({exc.__class__.__name__}: "
                      f"{str(exc)[:120]})", file=sys.stderr)
    if not ok:
        print("\nOpenReview is not reachable from this machine. If you are in a "
              "sandbox with an egress proxy, this is a policy denial and no "
              "amount of retrying will fix it — run this script somewhere with "
              "direct network access.", file=sys.stderr)
    return ok


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


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--years", default="2020-2025",
                   help="e.g. 2024 | 2020-2023 | 2020,2024 (default 2020-2025)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    p.add_argument("--limit", type=int, default=None,
                   help="only fetch the first N submissions per year (smoke test)")
    p.add_argument("--reparse", action="store_true",
                   help="re-parse from cache; never touch the network")
    p.add_argument("--refresh", action="store_true",
                   help="re-download even if a cache file exists")
    p.add_argument("--census", action="store_true",
                   help="write a per-year content-field census and exit")
    p.add_argument("--check", action="store_true",
                   help="probe API reachability and exit")
    args = p.parse_args()

    if args.check:
        return 0 if check_connectivity() else 1

    years = parse_years(args.years)
    out_dir = pathlib.Path(args.out)
    summaries, censuses = [], {}

    for year in years:
        print(f"\n=== ICLR {year} (API v{2 if year >= V2_FIRST_YEAR else 1}) ===")
        cache = cache_path(out_dir, year)

        if cache.exists() and not args.refresh:
            print(f"  cache hit: {cache}")
            raw = read_cache(cache)
        elif args.reparse:
            print(f"  ! no cache at {cache} and --reparse forbids downloading; "
                  f"skipping", file=sys.stderr)
            continue
        else:
            t0 = time.time()
            try:
                raw = download_year(year, limit=args.limit)
            except Exception as exc:
                print(f"  ! ICLR {year} failed: {exc.__class__.__name__}: {exc}",
                      file=sys.stderr)
                continue
            write_cache(cache, raw)
            print(f"  cached {len(raw)} submissions -> {cache} "
                  f"({time.time() - t0:.0f}s)")

        if args.census:
            censuses[year] = field_census(raw)
            print(json.dumps(censuses[year], indent=2)[:2000])
            continue

        records = [parse_submission(note, year) for note in raw]
        path = out_dir / f"iclr{year}.jsonl"
        n = write_jsonl(path, records)
        summary = summarize(year, records)
        summaries.append(summary)
        print(f"  wrote {n} records -> {path}")
        print(f"  {summary['with_ratings']} papers with >=2 ratings, "
              f"{summary['spread_ge_3']} with spread >=3")

        no_text = summary["total_reviews"] - summary["reviews_with_text"]
        if summary["total_reviews"] == 0:
            print(f"  ! no Official_Review replies parsed for {year} — run "
                  f"--census on this year; the invitation name probably differs",
                  file=sys.stderr)
        elif no_text > 0.05 * summary["total_reviews"]:
            print(f"  ! {no_text} reviews parsed with empty text — run --census "
                  f"on this year and check NON_TEXT_FIELDS", file=sys.stderr)

    if args.census:
        path = out_dir / "field_census.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(censuses, indent=2))
        print(f"\nWrote {path}")
        return 0

    if summaries:
        print_report(summaries)
        path = out_dir / "fetch_summary.json"
        path.write_text(json.dumps(summaries, indent=2))
        print(f"\nWrote {path}")
        return 0

    print("\nNothing fetched.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
