"""Offline tests for the SNOR loader (scripts/load_snor.py).

Fixtures mirror the real SNOR v1 schema, taken from an inspection of the
actual 2.2 GB dump (DOI 10.5281/zenodo.15866613): papers carry `review_scores`
and `accepted`, while the review prose lives nested inside each comment's
`content` dict, whose key set changes by year exactly as ICLR's did.
"""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "load_snor.py"
_spec = importlib.util.spec_from_file_location("load_snor", _SCRIPT)
ls = importlib.util.module_from_spec(_spec)
sys.modules["load_snor"] = ls
_spec.loader.exec_module(ls)


def comment(content, is_review=True, numeric_rating=6.0, rating="6: Weak Accept"):
    return {"conference_id": "iclr2022", "paper_id": "p1", "comment_id": "c1",
            "signature": "(anonymous)", "content": content, "reply_to_id": "p1",
            "is_review": is_review, "rating": rating, "numeric_rating": numeric_rating,
            "confidence": "4: The reviewer is confident", "numeric_confidence": 4.0}


def test_split_conf_id():
    assert ls.split_conf_id("iclr2017") == ("iclr", 2017)
    assert ls.split_conf_id("neurips2024") == ("neurips", 2024)
    assert ls.split_conf_id("garbage") == ("garbage", None)


# --------------------------------------------------------------------------
# Reviews out of comments
# --------------------------------------------------------------------------

def test_review_text_extracted_from_nested_content():
    """The prose is inside `content`, not at the top level."""
    rec = ls.review_from_comment(comment({
        "title": "Official Review",
        "review": "The paper is well motivated and the experiments are thorough.",
        "rating": "6: Weak Accept", "confidence": "4: confident",
    }))
    assert "well motivated" in rec["text"]
    assert rec["rating"] == 6.0
    assert rec["confidence"] == 4.0
    # Scores must not leak into the prose.
    assert "rating" not in rec["text_fields"]
    assert "confidence" not in rec["text_fields"]


def test_each_era_content_shape_yields_text():
    shapes = [
        {"title": "R", "review": "prose " * 30},                       # 2020/2021
        {"summary_of_the_paper": "s " * 20, "main_review": "m " * 30,  # 2022/2023
         "summary_of_the_review": "r " * 10},
        {"summary": "s " * 20, "strengths": "st " * 20,                # 2024+
         "weaknesses": "w " * 20, "questions": "q " * 10},
    ]
    for content in shapes:
        rec = ls.review_from_comment(comment(content))
        assert rec is not None and len(rec["text"]) > 50


def test_numeric_rating_preferred_but_string_is_a_fallback():
    with_numeric = ls.review_from_comment(comment({"review": "prose " * 20},
                                                  numeric_rating=8.0))
    assert with_numeric["rating"] == 8.0
    fallback = ls.review_from_comment(comment({"review": "prose " * 20},
                                              numeric_rating=None,
                                              rating="3: Weak Reject"))
    assert fallback["rating"] == 3.0


def test_comment_without_text_is_dropped():
    assert ls.review_from_comment(comment({"rating": "6: Weak Accept"})) is None


def test_unrated_review_still_kept_with_null_rating():
    """Text without a score is kept; build_pairs drops it if unusable."""
    rec = ls.review_from_comment(comment({"review": "prose " * 30},
                                         numeric_rating=None, rating=None))
    assert rec is not None and rec["rating"] is None


# --------------------------------------------------------------------------
# Papers
# --------------------------------------------------------------------------

def matched_paper(**kw):
    rec = {"id": "p1", "raw_decision": "ICLR 2022 Poster", "normalized_decision": "Poster",
           "title": "T", "abstract": "A", "keywords": ["dl"], "accepted": True,
           "conf_id": "iclr2022", "citation_count": 5}
    rec.update(kw)
    return rec


def test_matched_paper_record():
    rec = ls.paper_record(matched_paper(), "matched")
    assert rec["paper_id"] == "p1" and rec["year"] == 2022
    assert rec["conference"] == "iclr"
    assert rec["accepted"] is True
    assert rec["decision"] == "ICLR 2022 Poster"
    assert rec["decision_source"] == "snor_matched"
    assert rec["snor_source"] == "matched"


def test_snor_has_no_primary_area():
    """H3's area split cannot come from SNOR; keywords are kept instead."""
    rec = ls.paper_record(matched_paper(), "matched")
    assert rec["primary_area"] is None
    assert rec["keywords"] == ["dl"]


def test_rejected_and_withdrawn_papers():
    rejected = ls.paper_record(
        matched_paper(accepted=False, raw_decision="ICLR 2022 Rejected"), "matched")
    assert rejected["accepted"] is False and rejected["withdrawn"] is False
    withdrawn = ls.paper_record(
        matched_paper(accepted=False, raw_decision="ICLR 2022 Withdrawn"), "matched")
    assert withdrawn["withdrawn"] is True


def test_failed_match_paper_uses_venueid():
    """Unmatched papers keep venueid instead of raw_decision."""
    rec = ls.paper_record({"id": "f1", "venueid": "ICLR 2025 Rejected", "title": "T",
                           "abstract": "A", "keywords": [], "accepted": False,
                           "conference": "iclr2025"}, "failed_match")
    assert rec["year"] == 2025 and rec["accepted"] is False
    assert rec["decision_source"] == "snor_failed_match"
    assert rec["snor_source"] == "failed_match"


def test_accepted_falls_back_to_the_venue_string():
    """If SNOR's boolean is absent, read the decision text."""
    rec = ls.paper_record({"id": "x", "venueid": "ICLR 2024 Poster", "title": "T",
                           "abstract": "A", "conference": "iclr2024"}, "failed_match")
    assert rec["accepted"] is True


def test_paper_without_a_parseable_year_is_skipped():
    assert ls.paper_record(matched_paper(conf_id="iclr"), "matched") is None


# --------------------------------------------------------------------------
# End to end into the Task 3 schema
# --------------------------------------------------------------------------

def test_record_feeds_build_pairs_unchanged():
    """A SNOR-derived record must satisfy build_pairs without adaptation."""
    bp_spec = importlib.util.spec_from_file_location(
        "build_pairs", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_pairs.py")
    bp = importlib.util.module_from_spec(bp_spec)
    bp_spec.loader.exec_module(bp)

    rec = ls.paper_record(matched_paper(accepted=False,
                                        raw_decision="ICLR 2022 Rejected"), "matched")
    rec["reviews"] = [
        ls.review_from_comment(comment({"review": "positive " * 40}, numeric_rating=8.0)),
        ls.review_from_comment(comment({"review": "negative " * 40}, numeric_rating=3.0)),
    ]
    rec["reviews"][1]["review_id"] = "c2"

    pair, reason = bp.make_pair(rec, 5.5)
    assert reason is None
    # Paper was rejected, so the LOW review is the one the institution followed.
    assert pair["chosen_rating"] == 3.0 and pair["rejected_rating"] == 8.0
    assert pair["accepted"] is False
    assert pair["prompt"] == "T\n\nA"
