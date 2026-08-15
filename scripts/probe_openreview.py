"""
ICLR / OpenReview structure probe  —  step 1 of the peer-review preference project.

Purpose: find out what the data ACTUALLY looks like before writing an extractor.
Touches no more than a few hundred forums. Read-only. No credentials needed.

Writes probe_output.json (full detail, for Claude) and prints a short summary.

Run this FIRST, before scripts/fetch_openreview.py. It samples ~120 papers per
year in a couple of minutes and tells you whether the invitation names, field
names and rating scales this project assumes are the ones the API actually
serves. The full fetch takes hours; find out here instead.

Fixed 2026-08-15 (verified against openreview-py 2.4.2 by signature inspection):
  * `get_all_notes()` does NOT accept `limit` — it paginates the entire venue.
    The original probe called `get_all_notes(invitation=..., limit=SAMPLE)`,
    which raises TypeError on both API versions: the v2 path reported every
    year as FAILED and the v1 path silently sampled zero papers. Sampling
    needs `get_notes(..., limit=...)`, which is what this version calls.
  * The v2 path fetched replies with one `get_all_notes(forum=...)` call per
    paper. `details="replies"` returns them inline with the submissions, so a
    120-paper probe costs one request rather than 121.
  * ICLR 2021 added to the v1 year list; it was missing.
"""
import json, sys, traceback
from collections import Counter

SAMPLE = 120          # submissions sampled per year
YEARS_V2 = [2024, 2025]                # api2.openreview.net
YEARS_V1 = [2020, 2021, 2022, 2023]    # api.openreview.net

report = {"errors": [], "years": {}}


def note_content(note):
    """v1 stores content as {k: v}; v2 as {k: {'value': v}}. Normalize."""
    c = getattr(note, "content", {}) or {}
    if not isinstance(c, dict):
        return {}
    out = {}
    for k, v in c.items():
        out[k] = v.get("value") if isinstance(v, dict) and "value" in v else v
    return out


def summarize_reviews(reviews):
    """Return (field names seen, rating-like values found)."""
    fields, ratings = Counter(), []
    for r in reviews:
        c = note_content(r)
        fields.update(c.keys())
        for key in ("rating", "recommendation", "review_rating", "overall_rating"):
            if key in c:
                ratings.append((key, c[key]))
                break
    return fields, ratings


def _as_note(d):
    """Wrap a reply dict so note_content() can read it like a Note object."""
    return type("N", (), {"content": d.get("content", {}) or {}})()


def probe_v2(year):
    import openreview
    c = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    inv = f"ICLR.cc/{year}/Conference/-/Submission"
    # get_notes (not get_all_notes) is the one that takes limit; details brings
    # the replies inline instead of one extra request per paper.
    subs = c.get_notes(invitation=inv, details="replies", limit=SAMPLE)
    info = {"api": "v2", "invitation": inv, "n_sampled": len(subs)}
    if not subs:
        return info
    venues, review_invs, split_count, rev_counts = Counter(), Counter(), 0, []
    field_union, rating_examples, decisions = Counter(), [], Counter()
    for s in subs[:SAMPLE]:
        cc = note_content(s)
        venues[str(cc.get("venue"))] += 1
        replies = (getattr(s, "details", None) or {}).get("replies", []) or []
        reviews, nums = [], []
        for r in replies:
            invs = r.get("invitations") or []
            for i in invs:
                review_invs[str(i).split("/")[-1]] += 1
            if any("Official_Review" in str(i) for i in invs):
                reviews.append(_as_note(r))
                cr = note_content(_as_note(r))
                for key in ("rating", "recommendation"):
                    if key in cr:
                        try:
                            nums.append(float(str(cr[key]).split(":")[0].strip()))
                        except Exception:
                            pass
                        break
            if any("Decision" in str(i) for i in invs):
                decisions[str(note_content(_as_note(r)).get("decision"))] += 1
        rev_counts.append(len(reviews))
        f, rt = summarize_reviews(reviews)
        field_union.update(f)
        rating_examples.extend(rt[:2])
        if nums and (max(nums) - min(nums)) >= 3:
            split_count += 1
    info.update(venue_values=dict(venues.most_common(12)),
                reply_invitation_suffixes=dict(review_invs.most_common(15)),
                review_content_fields=dict(field_union.most_common(25)),
                rating_examples=rating_examples[:10],
                decision_values=dict(decisions.most_common(10)),
                reviews_per_paper=dict(Counter(rev_counts).most_common()),
                papers_with_rating_spread_ge_3=split_count,
                split_rate=round(split_count / max(1, len(subs)), 3))
    return info


def probe_v1(year):
    import openreview
    c = openreview.Client(baseurl="https://api.openreview.net")
    tried, subs, used = [], [], None
    for inv in (f"ICLR.cc/{year}/Conference/-/Blind_Submission",
                f"ICLR.cc/{year}/Conference/-/Submission"):
        try:
            got = c.get_notes(invitation=inv, details="directReplies", limit=SAMPLE)
            tried.append((inv, len(got)))
            if got:
                subs, used = got, inv
                break
        except Exception as e:
            tried.append((inv, f"ERR {e}"))
    info = {"api": "v1", "invitations_tried": tried, "invitation_used": used,
            "n_sampled": len(subs)}
    if not subs:
        return info
    venues, reply_invs, split_count, rev_counts = Counter(), Counter(), 0, []
    field_union, rating_examples, decisions = Counter(), [], Counter()
    for s in subs[:SAMPLE]:
        cc = note_content(s)
        venues[str(cc.get("venue") or cc.get("venueid"))] += 1
        replies = (getattr(s, "details", None) or {}).get("directReplies", []) or []
        reviews, nums = [], []
        for r in replies:
            inv = str(r.get("invitation", ""))
            reply_invs[inv.split("/")[-1]] += 1
            rc = r.get("content", {}) or {}
            if "Official_Review" in inv or "review" in rc:
                reviews.append(_as_note(r))
                for key in ("rating", "recommendation"):
                    if key in rc:
                        try:
                            nums.append(float(str(rc[key]).split(":")[0].strip()))
                        except Exception:
                            pass
                        break
            if "Decision" in inv:
                decisions[str(rc.get("decision"))] += 1
        rev_counts.append(len(reviews))
        f, rt = summarize_reviews(reviews)
        field_union.update(f)
        rating_examples.extend(rt[:2])
        if nums and (max(nums) - min(nums)) >= 3:
            split_count += 1
    info.update(venue_values=dict(venues.most_common(12)),
                reply_invitation_suffixes=dict(reply_invs.most_common(15)),
                review_content_fields=dict(field_union.most_common(25)),
                rating_examples=rating_examples[:10],
                decision_values=dict(decisions.most_common(10)),
                reviews_per_paper=dict(Counter(rev_counts).most_common()),
                papers_with_rating_spread_ge_3=split_count,
                split_rate=round(split_count / max(1, len(subs)), 3))
    return info


for y in YEARS_V2:
    try:
        report["years"][str(y)] = probe_v2(y)
        print(f"  ICLR {y} (v2): ok")
    except Exception as e:
        report["years"][str(y)] = {"fatal": str(e)}
        report["errors"].append(f"{y}: {traceback.format_exc(limit=2)}")
        print(f"  ICLR {y} (v2): FAILED - {e}")

for y in YEARS_V1:
    try:
        report["years"][str(y)] = probe_v1(y)
        print(f"  ICLR {y} (v1): ok")
    except Exception as e:
        report["years"][str(y)] = {"fatal": str(e)}
        report["errors"].append(f"{y}: {traceback.format_exc(limit=2)}")
        print(f"  ICLR {y} (v1): FAILED - {e}")

with open("probe_output.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1, ensure_ascii=False, default=str)

print("\n=========== SUMMARY (paste this back) ===========")
for y, info in report["years"].items():
    if "fatal" in info:
        print(f"ICLR {y}: FATAL {info['fatal'][:120]}")
        continue
    print(f"ICLR {y} [{info.get('api')}] sampled={info.get('n_sampled')} "
          f"split_rate={info.get('split_rate')} "
          f"reviews/paper={info.get('reviews_per_paper')}")
    print(f"   review fields: {list(info.get('review_content_fields', {}))[:12]}")
    print(f"   venue values : {list(info.get('venue_values', {}))[:6]}")
    if info.get("decision_values"):
        print(f"   decisions    : {info['decision_values']}")
print(f"errors: {len(report['errors'])}")
print("=================================================")
print("\nFull detail written to probe_output.json (send me that file too).")
