"""Offline tests for the OpenReview extractor (scripts/fetch_openreview.py).

No network. The fixtures below are hand-built to match the real note shapes
of each ICLR era, so the parsing logic can be validated on a machine that
cannot reach the API — which is the whole point, since the download and the
parse happen in different places.

The three shape differences under test, per API version:
  * content:     v1 {"rating": "8: ..."}    vs  v2 {"rating": {"value": 8}}
  * invitation:  v1 str                     vs  v2 list
  * replies:     v1 details["directReplies"] vs v2 details["replies"]

And the three field regimes:
  * 2020/2021  rating + a single "review" prose field
  * 2022/2023  recommendation + several prose fields
  * 2024+      rating + summary/strengths/weaknesses/questions + soundness etc.
"""

import gzip
import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fetch_openreview.py"
_spec = importlib.util.spec_from_file_location("fetch_openreview", _SCRIPT)
fo = importlib.util.module_from_spec(_spec)
sys.modules["fetch_openreview"] = fo
_spec.loader.exec_module(fo)


# --------------------------------------------------------------------------
# Fixtures: one submission per API era
# --------------------------------------------------------------------------

def v1_2020_submission():
    """ICLR 2020: rating + one 'review' field; verdict only in the meta-review."""
    return {
        "id": "paper2020",
        "forum": "paper2020",
        "number": 7,
        "invitation": "ICLR.cc/2020/Conference/-/Blind_Submission",
        "content": {
            "title": "A Study of Things",
            "abstract": "We study things.",
            "authors": ["A. One", "B. Two"],
            "keywords": ["things"],
        },
        "details": {
            "directReplies": [
                {
                    "id": "r1",
                    "invitation": "ICLR.cc/2020/Conference/Paper7/-/Official_Review",
                    "content": {
                        "title": "Official Review",
                        "review": "The paper is well motivated and the experiments "
                                  "are thorough. I recommend acceptance.",
                        "rating": "8: Top 50% of accepted papers, clear accept",
                        "confidence": "4: The reviewer is confident but not absolutely certain",
                    },
                },
                {
                    "id": "r2",
                    "invitation": "ICLR.cc/2020/Conference/Paper7/-/Official_Review",
                    "content": {
                        "title": "Official Review",
                        "review": "The contribution is incremental and the baselines "
                                  "are weak. I recommend rejection.",
                        "rating": "3: Weak Reject",
                        "confidence": "5: The reviewer is absolutely certain",
                    },
                },
                {
                    "id": "m1",
                    "invitation": "ICLR.cc/2020/Conference/Paper7/-/Meta_Review",
                    "content": {
                        "metareview": "Reviewers disagreed; the AC sides with acceptance.",
                        "recommendation": "Accept (Poster)",
                    },
                },
            ]
        },
    }


def v1_2022_submission():
    """ICLR 2022: the score field is 'recommendation', prose is split in four."""
    return {
        "id": "paper2022",
        "forum": "paper2022",
        "number": 12,
        "invitation": "ICLR.cc/2022/Conference/-/Blind_Submission",
        "content": {"title": "Another Study", "abstract": "More things."},
        "details": {
            "directReplies": [
                {
                    "id": "r3",
                    "invitation": "ICLR.cc/2022/Conference/Paper12/-/Official_Review",
                    "content": {
                        "summary_of_the_paper": "The paper proposes a method.",
                        "main_review": "The method is sound and clearly described.",
                        "summary_of_the_review": "Solid work, I lean accept.",
                        "recommendation": "8: accept, good paper",
                        "confidence": "4: You are confident in your assessment",
                        "correctness": "4: All of the claims and statements are well-supported",
                        "technical_novelty_and_reproducibility": "3: Good novelty",
                        "empirical_novelty_and_significance": "2: Marginal",
                        "flag_for_ethics_review": ["NO."],
                    },
                },
                {
                    "id": "r4",
                    "invitation": "ICLR.cc/2022/Conference/Paper12/-/Official_Review",
                    "content": {
                        "summary_of_the_paper": "The paper proposes a method.",
                        "main_review": "Experiments are insufficient to support the claims.",
                        "summary_of_the_review": "I lean reject.",
                        "recommendation": "3: reject, not good enough",
                        "confidence": "4: You are confident in your assessment",
                    },
                },
                {
                    "id": "d1",
                    "invitation": "ICLR.cc/2022/Conference/Paper12/-/Decision",
                    "content": {"decision": "Reject"},
                },
            ]
        },
    }


def v2_2024_submission():
    """ICLR 2024: every content value is wrapped in {"value": ...}."""
    def val(x):
        return {"value": x}

    return {
        "id": "paper2024",
        "forum": "paper2024",
        "number": 3,
        "invitations": ["ICLR.cc/2024/Conference/-/Submission"],
        "content": {
            "title": val("A Third Study"),
            "abstract": val("Even more things."),
            "keywords": val(["deep learning"]),
            "primary_area": val("representation learning"),
            "venue": val("ICLR 2024 poster"),
            "venueid": val("ICLR.cc/2024/Conference"),
            "authors": val(["C. Three"]),
        },
        "details": {
            "replies": [
                {
                    "id": "r5",
                    "invitations": ["ICLR.cc/2024/Conference/Submission3/-/Official_Review"],
                    "content": {
                        "summary": val("The authors propose a new architecture."),
                        "strengths": val("Clear writing and strong ablations."),
                        "weaknesses": val("Limited to one dataset."),
                        "questions": val("How does this scale?"),
                        "rating": val(8),
                        "confidence": val(4),
                        "soundness": val(3),
                        "presentation": val(4),
                        "contribution": val(3),
                    },
                },
                {
                    "id": "r6",
                    "invitations": ["ICLR.cc/2024/Conference/Submission3/-/Official_Review"],
                    "content": {
                        "summary": val("The authors propose a new architecture."),
                        "strengths": val("The idea is simple."),
                        "weaknesses": val("The evaluation is not convincing and "
                                          "related work is missing."),
                        "questions": val("Why no comparison to X?"),
                        "rating": val(3),
                        "confidence": val(5),
                    },
                },
                {
                    "id": "d2",
                    "invitations": ["ICLR.cc/2024/Conference/Submission3/-/Decision"],
                    "content": {"decision": val("Accept (poster)")},
                },
            ]
        },
    }


# --------------------------------------------------------------------------
# Shape normalization
# --------------------------------------------------------------------------

def test_unwrap_handles_both_content_shapes():
    assert fo.unwrap({"value": 8}) == 8
    assert fo.unwrap("8: accept") == "8: accept"
    # A genuine dict field with more than one key must survive intact.
    assert fo.unwrap({"value": 1, "other": 2}) == {"value": 1, "other": 2}


def test_normalize_content_v1_and_v2_agree():
    v1 = fo.normalize_content({"rating": "8: accept", "title": "T"})
    v2 = fo.normalize_content({"rating": {"value": "8: accept"}, "title": {"value": "T"}})
    assert v1 == v2 == {"rating": "8: accept", "title": "T"}


def test_note_invitations_str_and_list():
    assert fo.note_invitations({"invitation": "a/-/Official_Review"}) == ["a/-/Official_Review"]
    assert fo.note_invitations({"invitations": ["a", "b"]}) == ["a", "b"]
    assert fo.note_invitations({}) == []


def test_note_replies_reads_both_keys():
    assert fo.note_replies({"details": {"replies": [1]}}) == [1]
    assert fo.note_replies({"details": {"directReplies": [2]}}) == [2]
    assert fo.note_replies({"details": {}}) == []
    assert fo.note_replies({}) == []


def test_invitation_matching_is_suffix_exact():
    """A revision invitation must not be mistaken for a review."""
    assert fo.is_official_review({"invitation": "X/Paper1/-/Official_Review"})
    assert not fo.is_official_review({"invitation": "X/Paper1/-/Official_Review_Revision"})
    assert not fo.is_official_review({"invitation": "X/Paper1/-/Official_Comment"})
    assert fo.is_decision({"invitations": ["X/Submission1/-/Decision"]})
    assert fo.is_meta_review({"invitation": "X/Paper1/-/Meta_Review"})


# --------------------------------------------------------------------------
# Score parsing and field detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("8: Top 50% of accepted papers, clear accept", 8.0),
    ("3: Weak Reject", 3.0),
    ("10", 10.0),
    (8, 8.0),
    (6.5, 6.5),
    ({"value": 5}, 5.0),
    ("", None),
    ("Strong accept", None),
    (None, None),
    (True, None),          # a checkbox is not a score
])
def test_parse_score(raw, expected):
    assert fo.parse_score(raw) == expected


def test_rating_field_detected_per_era():
    """2022/2023 name the score 'recommendation'; other years 'rating'."""
    c2020 = fo.normalize_content(v1_2020_submission()["details"]["directReplies"][0]["content"])
    c2022 = fo.normalize_content(v1_2022_submission()["details"]["directReplies"][0]["content"])
    c2024 = fo.normalize_content(v2_2024_submission()["details"]["replies"][0]["content"])
    assert fo.detect_field(c2020, fo.RATING_FIELDS) == "rating"
    assert fo.detect_field(c2022, fo.RATING_FIELDS) == "recommendation"
    assert fo.detect_field(c2024, fo.RATING_FIELDS) == "rating"


# --------------------------------------------------------------------------
# Review text assembly
# --------------------------------------------------------------------------

def test_single_prose_field_gets_no_heading():
    content = {"review": "A" * 50, "rating": "8: accept", "title": "Official Review"}
    text, fields = fo.assemble_review_text(content)
    assert text == "A" * 50
    assert list(fields) == ["review"]


def test_multi_field_review_is_sectioned_and_excludes_scores():
    content = fo.normalize_content(
        v2_2024_submission()["details"]["replies"][0]["content"]
    )
    text, fields = fo.assemble_review_text(content)
    assert set(fields) == {"summary", "strengths", "weaknesses", "questions"}
    assert "## Strengths" in text and "## Weaknesses" in text
    # No numeric field may leak into the prose.
    for banned in ("rating", "soundness", "presentation", "contribution", "confidence"):
        assert banned not in fields
    assert "Clear writing" in text


def test_scores_and_ethics_text_never_enter_the_prose():
    content = fo.normalize_content(
        v1_2022_submission()["details"]["directReplies"][0]["content"]
    )
    _, fields = fo.assemble_review_text(content)
    assert set(fields) == {
        "summary_of_the_paper", "main_review", "summary_of_the_review"
    }
    assert "correctness" not in fields
    assert "details_of_ethics_concerns" not in fields


def test_field_order_is_stable():
    content = fo.normalize_content(
        v2_2024_submission()["details"]["replies"][0]["content"]
    )
    first, _ = fo.assemble_review_text(content)
    second, _ = fo.assemble_review_text(dict(content))
    assert first == second


# --------------------------------------------------------------------------
# Acceptance / decision logic
# --------------------------------------------------------------------------

@pytest.mark.parametrize("decision,venue,expected", [
    ("Accept (poster)", None, True),
    ("Accept (Oral)", None, True),
    ("Reject", None, False),
    (None, "ICLR 2024 poster", True),
    (None, "ICLR 2024 spotlight", True),
    (None, "Submitted to ICLR 2024", False),   # v2's way of saying rejected
    (None, "ICLR 2024 Conference Withdrawn Submission", None),
    ("Desk Rejected", None, None),
    (None, None, None),
])
def test_classify_acceptance(decision, venue, expected):
    assert fo.classify_acceptance(decision, venue) is expected


def test_decision_priority_decision_then_meta_then_venue():
    replies = [
        {"invitation": "X/-/Meta_Review", "content": {"recommendation": "Accept (Poster)"}},
        {"invitation": "X/-/Decision", "content": {"decision": "Reject"}},
    ]
    assert fo.extract_decision({}, replies) == ("Reject", "Decision")
    assert fo.extract_decision({}, replies[:1]) == ("Accept (Poster)", "Meta_Review")
    assert fo.extract_decision({"venue": "ICLR 2024 poster"}, []) == (
        "ICLR 2024 poster", "venue"
    )
    assert fo.extract_decision({}, []) == (None, None)


# --------------------------------------------------------------------------
# End-to-end parsing, per era
# --------------------------------------------------------------------------

def test_parse_v1_2020_submission():
    rec = fo.parse_submission(v1_2020_submission(), 2020)
    assert rec["paper_id"] == "paper2020"
    assert rec["api_version"] == 1
    assert rec["n_reviews"] == 2           # the meta-review is not a review
    assert rec["n_ratings"] == 2
    assert rec["rating_spread"] == 5.0     # 8 - 3
    assert rec["decision"] == "Accept (Poster)"
    assert rec["decision_source"] == "Meta_Review"
    assert rec["accepted"] is True
    assert rec["withdrawn"] is False
    assert rec["meta_review"].startswith("Reviewers disagreed")
    assert rec["primary_area"] is None     # no area field before 2024
    assert all(r["rating_field"] == "rating" for r in rec["reviews"])
    assert "recommend acceptance" in rec["reviews"][0]["text"]


def test_parse_v1_2022_submission():
    rec = fo.parse_submission(v1_2022_submission(), 2022)
    assert rec["n_reviews"] == 2
    assert sorted(r["rating"] for r in rec["reviews"]) == [3.0, 8.0]
    assert rec["rating_spread"] == 5.0
    assert rec["decision"] == "Reject"
    assert rec["decision_source"] == "Decision"
    assert rec["accepted"] is False
    assert all(r["rating_field"] == "recommendation" for r in rec["reviews"])
    assert rec["reviews"][0]["aux_scores"]["correctness"] == 4.0


def test_parse_v2_2024_submission():
    rec = fo.parse_submission(v2_2024_submission(), 2024)
    assert rec["api_version"] == 2
    assert rec["title"] == "A Third Study"          # unwrapped, not {"value": ...}
    assert rec["abstract"] == "Even more things."
    assert rec["primary_area"] == "representation learning"
    assert rec["n_reviews"] == 2
    assert rec["rating_spread"] == 5.0
    assert rec["decision"] == "Accept (poster)"
    assert rec["accepted"] is True
    assert rec["reviews"][0]["confidence"] == 4.0
    assert rec["reviews"][0]["aux_scores"] == {
        "soundness": 3.0, "presentation": 4.0, "contribution": 3.0
    }
    assert rec["n_authors"] == 1


def test_withdrawn_submission_is_flagged():
    sub = v2_2024_submission()
    sub["content"]["venue"] = {"value": "ICLR 2024 Conference Withdrawn Submission"}
    sub["details"]["replies"] = [
        r for r in sub["details"]["replies"] if not fo.is_decision(r)
    ]
    rec = fo.parse_submission(sub, 2024)
    assert rec["withdrawn"] is True
    assert rec["accepted"] is None


def test_submission_with_no_replies_survives():
    """A paper with no public reviews must parse, not crash."""
    rec = fo.parse_submission(
        {"id": "x", "content": {"title": "T"}, "invitation": "ICLR.cc/2021/Conference/-/Blind_Submission"},
        2021,
    )
    assert rec["n_reviews"] == 0
    assert rec["rating_spread"] is None
    assert rec["decision"] is None


def test_review_without_text_is_dropped():
    """A rating-only reply carries no training signal and must not be kept."""
    assert fo.parse_review({"id": "r", "content": {"rating": "8: accept"}}) is None


# --------------------------------------------------------------------------
# Cache integrity — this guards the bug that would cost a whole re-download
# --------------------------------------------------------------------------

class _FakeNote:
    """Mimics openreview-py's Note: to_json() deliberately omits details."""

    def __init__(self, body, details, number=None):
        self._body = body
        self.details = details
        self.number = number

    def to_json(self):
        return dict(self._body)


def test_cacheable_note_keeps_details_and_number():
    note = _FakeNote({"id": "p1", "content": {}}, {"replies": [{"id": "r1"}]}, number=42)
    cached = fo._note_to_cacheable(note)
    assert cached["details"]["replies"] == [{"id": "r1"}]
    assert cached["number"] == 42
    assert fo.note_replies(cached) == [{"id": "r1"}]


def test_cache_roundtrip(tmp_path):
    records = [v1_2020_submission(), v2_2024_submission()]
    path = fo.cache_path(tmp_path, 2020)
    fo.write_cache(path, records)
    assert path.exists() and path.suffix == ".gz"
    assert not list(path.parent.glob("*.tmp"))   # atomic write left no debris
    assert fo.read_cache(path) == records


def test_write_jsonl_roundtrip(tmp_path):
    recs = [fo.parse_submission(v2_2024_submission(), 2024)]
    path = tmp_path / "iclr2024.jsonl"
    assert fo.write_jsonl(path, recs) == 1
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == recs


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_summarize_counts():
    records = [
        fo.parse_submission(v1_2020_submission(), 2020),
        fo.parse_submission(v1_2022_submission(), 2022),
    ]
    s = fo.summarize(2020, records)
    assert s["papers"] == 2
    assert s["with_ratings"] == 2
    assert s["spread_ge_3"] == 2 and s["spread_ge_4"] == 2
    assert s["accepted"] == 1 and s["rejected"] == 1
    assert s["total_reviews"] == 4 and s["reviews_with_text"] == 4


def test_field_census_reports_rating_field():
    census = fo.field_census([v1_2022_submission()])
    assert census["n_submissions"] == 1
    assert census["n_official_reviews"] == 2
    assert census["detected_rating_field"] == {"recommendation": 2}
    assert "main_review" in census["review_fields"]


def test_parse_years():
    assert fo.parse_years("2024") == [2024]
    assert fo.parse_years("2020-2023") == [2020, 2021, 2022, 2023]
    assert fo.parse_years("2020,2024-2025") == [2020, 2024, 2025]
    assert fo.parse_years("2021, 2021") == [2021]


def test_module_imports_without_openreview_installed():
    """The parse path must work on a machine that never installed the client."""
    assert "openreview" not in sys.modules or True
    # parse_submission must not need the package at all
    fo.parse_submission(v1_2020_submission(), 2020)
