"""Offline tests for pair construction (scripts/build_pairs.py).

The construction rule is the whole experiment, so it is tested directly:
which review becomes `chosen`, when a paper yields no pair at all, and how
the against-the-majority subset is identified.

Two properties matter more than the rest and have dedicated tests:

  * `chosen` follows the *institution*, not the higher rating. On a rejected
    paper the low-rated review is chosen. Getting this backwards would invert
    the labels on 71% of the dataset and still look plausible in aggregate.
  * ties are broken by review id, never by length. Preferring the longer
    review would build the length confound directly into the labels that H2
    exists to rule out.
"""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_pairs.py"
_spec = importlib.util.spec_from_file_location("build_pairs", _SCRIPT)
bp = importlib.util.module_from_spec(_spec)
sys.modules["build_pairs"] = bp
_spec.loader.exec_module(bp)


def review(rid, rating, text=None, confidence=4.0):
    return {"review_id": rid, "rating": rating, "confidence": confidence,
            "text": text if text is not None else f"Review {rid}. " + "word " * 20,
            "text_fields": {}}


def paper(reviews, accepted=True, year=2024, **kw):
    record = {
        "paper_id": kw.pop("paper_id", "p1"),
        "year": year,
        "title": "A Title",
        "abstract": "An abstract.",
        "primary_area": "representation learning",
        "decision": "Accept (poster)" if accepted else "Reject",
        "decision_source": "Decision",
        "accepted": accepted,
        "withdrawn": False,
        "desk_rejected": False,
        "reviews": reviews,
    }
    record.update(kw)
    return record


# --------------------------------------------------------------------------
# Threshold estimation
# --------------------------------------------------------------------------

def test_threshold_separates_accepted_from_rejected():
    """A clean dataset: everything >= 6 accepted, everything <= 4 rejected."""
    records = []
    for i in range(30):
        records.append(paper([review(f"a{i}", 8), review(f"b{i}", 6)],
                             accepted=True, paper_id=f"acc{i}"))
    for i in range(30):
        records.append(paper([review(f"c{i}", 4), review(f"d{i}", 2)],
                             accepted=False, paper_id=f"rej{i}"))
    est = bp.estimate_threshold(records)
    assert 4.0 < est["threshold"] <= 7.0
    assert est["youden_j"] == pytest.approx(1.0)
    assert est["sensitivity"] == 1.0 and est["specificity"] == 1.0


def test_threshold_adapts_to_a_different_rating_scale():
    """ICLR 2020 rates on {1,3,6,8}; a hardcoded 5 would be meaningless."""
    records = []
    for i in range(25):
        records.append(paper([review(f"a{i}", 8), review(f"b{i}", 6)],
                             accepted=True, paper_id=f"acc{i}"))
    for i in range(25):
        records.append(paper([review(f"c{i}", 3), review(f"d{i}", 1)],
                             accepted=False, paper_id=f"rej{i}"))
    est = bp.estimate_threshold(records)
    assert 3.0 < est["threshold"] <= 7.0


def test_threshold_reports_when_it_cannot_be_estimated():
    assert bp.estimate_threshold([])["threshold"] is None
    only_accepted = [paper([review("a", 8), review("b", 6)], paper_id=f"p{i}")
                     for i in range(30)]
    est = bp.estimate_threshold(only_accepted)
    assert est["threshold"] is None
    assert "one outcome class" in est["note"]


# --------------------------------------------------------------------------
# The construction rule
# --------------------------------------------------------------------------

def test_accepted_paper_chooses_the_positive_review():
    pair, reason = bp.make_pair(paper([review("hi", 8), review("lo", 3)]), 5.0)
    assert reason is None
    assert pair["chosen_rating"] == 8 and pair["rejected_rating"] == 3
    assert pair["chosen_review_id"] == "hi"
    assert pair["accepted"] is True


def test_rejected_paper_chooses_the_negative_review():
    """The institution is the arbiter, so the LOW review is chosen here."""
    pair, reason = bp.make_pair(
        paper([review("hi", 8), review("lo", 3)], accepted=False), 5.0
    )
    assert reason is None
    assert pair["chosen_rating"] == 3, "chosen must follow the decision, not the rating"
    assert pair["rejected_rating"] == 8
    assert pair["chosen_review_id"] == "lo"
    assert pair["accepted"] is False


def test_most_extreme_pair_is_taken_and_alternatives_counted():
    reviews = [review("a", 9), review("b", 6), review("c", 4), review("d", 1)]
    pair, _ = bp.make_pair(paper(reviews), 5.0)
    assert pair["chosen_rating"] == 9 and pair["rejected_rating"] == 1
    assert pair["n_reviews"] == 4
    assert pair["n_discarded_alternatives"] == 2


def test_ties_break_on_id_not_on_length():
    """Two reviews tie at 8; the longer one must not win by being longer."""
    short = review("aaa", 8, text="short " * 10)
    long = review("zzz", 8, text="long " * 400)
    pair, _ = bp.make_pair(paper([long, short, review("r", 2)]), 5.0)
    assert pair["chosen_review_id"] == "aaa", "tie must break on id, never on length"


def test_prompt_is_title_and_abstract():
    pair, _ = bp.make_pair(paper([review("a", 8), review("b", 2)]), 5.0)
    assert pair["prompt"] == "A Title\n\nAn abstract."


def test_lengths_are_recorded():
    pair, _ = bp.make_pair(
        paper([review("a", 8, text="one two three"), review("b", 2, text="one two")]), 5.0
    )
    assert pair["chosen_len_tokens"] == 3
    assert pair["rejected_len_tokens"] == 2


# --------------------------------------------------------------------------
# The against-the-majority subset
# --------------------------------------------------------------------------

def test_majority_agreed_when_decision_follows_the_crowd():
    reviews = [review("a", 8), review("b", 7), review("c", 2)]
    pair, _ = bp.make_pair(paper(reviews, accepted=True), 5.0)
    assert pair["majority_agreed_with_decision"] is True
    assert pair["n_accept_side"] == 2 and pair["n_reject_side"] == 1


def test_majority_disagreed_is_the_honest_test_subset():
    """Two reviewers said reject, the venue accepted anyway."""
    reviews = [review("a", 8), review("b", 3), review("c", 2)]
    pair, _ = bp.make_pair(paper(reviews, accepted=True), 5.0)
    assert pair["majority_agreed_with_decision"] is False


def test_even_split_has_no_majority():
    reviews = [review("a", 8), review("b", 7), review("c", 3), review("d", 2)]
    pair, _ = bp.make_pair(paper(reviews, accepted=True), 5.0)
    assert pair["majority_agreed_with_decision"] is None


# --------------------------------------------------------------------------
# Exclusions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("record,expected", [
    (paper([review("a", 8), review("b", 3)], year=2026), "excluded_year"),
    (paper([review("a", 8), review("b", 3)], withdrawn=True), "withdrawn_or_desk_rejected"),
    (paper([review("a", 8), review("b", 3)], desk_rejected=True), "withdrawn_or_desk_rejected"),
    (paper([review("a", 8), review("b", 3)], accepted=None), "no_decision"),
    (paper([review("a", 8)]), "too_few_rated_reviews"),
    (paper([review("a", 8), review("b", 7)]), "no_straddle"),
    (paper([review("a", 8), review("b", 3, text="")]), "missing_text"),
])
def test_exclusion_reasons(record, expected):
    pair, reason = bp.make_pair(record, 5.0)
    assert pair is None and reason == expected


def test_agreeing_reviewers_yield_no_pair():
    """Both reviewers on the same side means no disagreement to arbitrate."""
    _, reason = bp.make_pair(paper([review("a", 8), review("b", 9)]), 5.0)
    assert reason == "no_straddle"
    _, reason = bp.make_pair(paper([review("a", 2), review("b", 1)], accepted=False), 5.0)
    assert reason == "no_straddle"


def test_min_spread_filter():
    record = paper([review("a", 6), review("b", 4)])
    assert bp.make_pair(record, 5.0, min_spread=1.0)[0] is not None
    assert bp.make_pair(record, 5.0, min_spread=3.0)[1] == "spread_below_min"


def test_identical_text_is_dropped():
    same = "the same review text repeated"
    record = paper([review("a", 8, text=same), review("b", 3, text=same)])
    assert bp.make_pair(record, 5.0)[1] == "identical_text"


def test_missing_title_and_abstract_is_dropped():
    record = paper([review("a", 8), review("b", 3)], title="", abstract="")
    assert bp.make_pair(record, 5.0)[1] == "no_prompt"


def test_every_exclusion_reason_is_declared():
    """Any reason make_pair can return must be listed for the dataset card."""
    records = [
        paper([review("a", 8), review("b", 3)], year=2026),
        paper([review("a", 8), review("b", 3)], withdrawn=True),
        paper([review("a", 8), review("b", 3)], accepted=None),
        paper([review("a", 8)]),
        paper([review("a", 8), review("b", 7)]),
        paper([review("a", 8), review("b", 3, text="")]),
        paper([review("a", 8), review("b", 3)], title="", abstract=""),
    ]
    for record in records:
        _, reason = bp.make_pair(record, 5.0)
        assert reason in bp.EXCLUSION_REASONS


# --------------------------------------------------------------------------
# Output schema and files
# --------------------------------------------------------------------------

REQUIRED_SCHEMA = (
    "paper_id", "year", "primary_area", "prompt", "chosen", "rejected",
    "chosen_rating", "rejected_rating", "decision", "accepted",
    "majority_agreed_with_decision", "chosen_len_tokens", "rejected_len_tokens",
)


def test_pair_has_the_handoff_schema():
    pair, _ = bp.make_pair(paper([review("a", 8), review("b", 3)]), 5.0)
    for field in REQUIRED_SCHEMA:
        assert field in pair, f"missing required field {field}"


def test_index_contains_no_review_text(tmp_path):
    """The released artifact must carry identifiers, never prose (HANDOFF s8)."""
    pairs = [bp.make_pair(paper([review("a", 8, text="SECRET REVIEW PROSE"),
                                 review("b", 3)]), 5.0)[0]]
    path = tmp_path / "pairs_index.csv"
    bp.write_index(path, pairs)
    body = path.read_text(encoding="utf-8")
    assert "SECRET REVIEW PROSE" not in body
    assert "a" in body and "8" in body
    # The text-bearing columns must not be in the released schema at all.
    assert "chosen" not in bp.INDEX_COLUMNS
    assert "rejected" not in bp.INDEX_COLUMNS
    assert "prompt" not in bp.INDEX_COLUMNS


def test_jsonl_roundtrip(tmp_path):
    import json
    pairs = [bp.make_pair(paper([review("a", 8), review("b", 3)]), 5.0)[0]]
    path = tmp_path / "pairs.jsonl"
    assert bp.write_jsonl(path, pairs) == 1
    assert [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] == pairs


def test_card_reports_the_traps(tmp_path):
    import collections
    pairs = []
    for i in range(10):
        p, _ = bp.make_pair(paper([review(f"a{i}", 8), review(f"b{i}", 3)],
                                  accepted=i % 2 == 0, paper_id=f"p{i}"), 5.0)
        pairs.append(p)
    card = bp.build_card(
        pairs,
        {2024: {"threshold": 5.0, "youden_j": 0.9, "sensitivity": 0.9, "specificity": 0.9}},
        {2024: collections.Counter({"no_straddle": 3})},
        {2024: 20}, 1.0, "whitespace word count",
    )
    assert "70%" in card                       # the imbalance trap is named
    assert "against" in card.lower()           # the honest-test subset is named
    assert "CC BY 4.0" in card                 # licence is stated
    assert "must not be used in a deployed reviewing system" in card
    assert "no_straddle" in card               # exclusions are accounted for


def test_parse_years():
    assert bp.parse_years("2020-2023") == [2020, 2021, 2022, 2023]
    assert bp.parse_years("2024") == [2024]
    assert bp.parse_years("2020,2024-2025") == [2020, 2024, 2025]
