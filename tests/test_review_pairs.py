"""Tests for the splits, arms and baselines (evolving_dpo.review_pairs).

The invariants here are the ones that decide whether a results table means
anything. Two matter most:

  * balancing must drive the always-negative baseline to exactly 50%. If it
    does not, every accuracy in the paper is inflated by the 71% rejection
    rate and the study reports a class prior as a finding.
  * the shuffled arm must never leave a review with its own paper, or the
    control that is supposed to destroy the institutional signal quietly
    preserves it.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from evolving_dpo import review_pairs as rp  # noqa: E402


def pair(pid, accepted=True, chosen_rating=8.0, rejected_rating=3.0,
         chosen="A thorough review. " * 20, rejected="A short review. " * 5,
         year=2024, keywords=None, **kw):
    p = {
        "paper_id": pid, "year": year, "primary_area": None,
        "keywords": keywords if keywords is not None else ["deep learning"],
        "prompt": f"Title {pid}\n\nAbstract for {pid}.",
        "chosen": chosen, "rejected": rejected,
        "chosen_rating": chosen_rating, "rejected_rating": rejected_rating,
        "accepted": accepted, "majority_agreed_with_decision": True,
        "chosen_len_tokens": len(chosen.split()),
        "rejected_len_tokens": len(rejected.split()),
    }
    p.update(kw)
    return p


def corpus(n_accepted=30, n_rejected=70):
    """A corpus with ICLR's real ~70/30 imbalance."""
    out = []
    for i in range(n_accepted):
        out.append(pair(f"acc{i}", accepted=True, chosen_rating=8.0, rejected_rating=3.0))
    for i in range(n_rejected):
        # For a rejected paper the chosen review is the NEGATIVE one.
        out.append(pair(f"rej{i}", accepted=False, chosen_rating=3.0, rejected_rating=8.0))
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_load_pairs_roundtrip(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [pair("a", year=2020), pair("b", year=2024)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(rp.load_pairs(path)) == 2
    assert [p["paper_id"] for p in rp.load_pairs(path, years=[2024])] == ["b"]


# --------------------------------------------------------------------------
# The 70% trap
# --------------------------------------------------------------------------

def test_always_negative_scores_the_rejection_rate_on_raw_data():
    """The trap, demonstrated: 70% without balancing."""
    assert rp.baseline_always_negative(corpus(30, 70)) == pytest.approx(0.70)


def test_balancing_drives_always_negative_to_exactly_half():
    """The mitigation, verified. This is the point of balancing."""
    balanced = rp.balance_accept_reject(corpus(30, 70), seed=0)
    assert len(balanced) == 60
    assert sum(1 for p in balanced if p["accepted"]) == 30
    assert rp.baseline_always_negative(balanced) == pytest.approx(0.50)


def test_balancing_is_deterministic_given_a_seed():
    a = rp.balance_accept_reject(corpus(), seed=7)
    b = rp.balance_accept_reject(corpus(), seed=7)
    c = rp.balance_accept_reject(corpus(), seed=8)
    assert [p["paper_id"] for p in a] == [p["paper_id"] for p in b]
    assert [p["paper_id"] for p in a] != [p["paper_id"] for p in c]


def test_balancing_an_already_balanced_corpus_keeps_everything():
    assert len(rp.balance_accept_reject(corpus(50, 50))) == 100


def test_always_longer_baseline():
    """chosen is longer in every pair here, so the baseline is 1.0."""
    assert rp.baseline_always_longer(corpus(10, 10)) == pytest.approx(1.0)
    mixed = [pair("a", chosen="x " * 5, rejected="y " * 50),
             pair("b", chosen="x " * 50, rejected="y " * 5)]
    assert rp.baseline_always_longer(mixed) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def test_temporal_split_is_structural():
    pairs = [pair(f"p{y}", year=y) for y in (2020, 2021, 2022, 2023, 2024, 2025)]
    train, test = rp.temporal_split(pairs)
    assert {p["year"] for p in train} == {2020, 2021, 2022, 2023}
    assert {p["year"] for p in test} == {2024, 2025}
    assert not ({p["paper_id"] for p in train} & {p["paper_id"] for p in test})


def test_topic_split_partitions_by_area():
    pairs, _ = rp.assign_areas([
        pair("a", keywords=["reinforcement learning", "policy gradient"]),
        pair("b", keywords=["graph neural network", "node classification"]),
    ])
    split = rp.topic_split(pairs, "reinforcement_learning", "graph")
    assert [p["paper_id"] for p in split["a"]] == ["a"]
    assert [p["paper_id"] for p in split["b"]] == ["b"]


def test_topic_split_rejects_one_area():
    with pytest.raises(ValueError):
        rp.topic_split([], "theory", "theory")


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------

def test_area_from_keywords():
    assert rp.assign_area(pair("x", keywords=["reinforcement learning"])) == "reinforcement_learning"
    assert rp.assign_area(pair("x", keywords=["graph neural network"])) == "graph"
    assert rp.assign_area(pair("x", keywords=["differential privacy"])) == "trustworthy"


def test_primary_area_is_preferred_when_present():
    p = pair("x", keywords=["nothing relevant"], primary_area="reinforcement learning")
    assert rp.assign_area(p) == "reinforcement_learning"


def test_unmatched_pair_gets_no_area():
    p = pair("x", keywords=["basket weaving"])
    p["prompt"] = "Title\n\nAbstract about basket weaving."
    assert rp.assign_area(p) is None


def test_area_assignment_is_deterministic():
    """Ties resolve the same way on every machine, or splits are irreproducible."""
    pairs = [pair(f"p{i}", keywords=["deep learning"]) for i in range(20)]
    first = [rp.assign_area(p) for p in pairs]
    second = [rp.assign_area(p) for p in pairs]
    assert first == second


def test_assign_areas_reports_coverage():
    pairs, report = rp.assign_areas([
        pair("a", keywords=["reinforcement learning"]),
        pair("b", keywords=["basket weaving"]),
    ])
    pairs[1]["prompt"] = "x"
    assert report["n_pairs"] == 2
    assert report["n_assigned"] + report["n_unassigned"] == 2
    assert "by_area" in report and 0.0 <= report["coverage"] <= 1.0


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def test_preference_examples_match_the_trainer_contract():
    ex = rp.to_preference_examples(corpus(2, 2))
    assert all(set(e) == {"prompt", "chosen", "rejected"} for e in ex)
    assert len(ex) == 4


def test_sft_examples_are_chosen_only():
    ex = rp.to_sft_examples(corpus(2, 2))
    assert all(set(e) == {"prompt", "completion"} for e in ex)
    assert ex[0]["completion"] == corpus(2, 2)[0]["chosen"]


def test_shuffled_pairs_never_keep_their_own_paper():
    """The control must actually destroy the paper-specific signal."""
    shuffled = rp.shuffled_pairs(corpus(10, 10), seed=0)
    assert len(shuffled) > 0
    for s in shuffled:
        assert s["paper_id"] != s["donor_paper_id"]


def test_shuffled_pairs_keep_the_chosen_review():
    original = {p["paper_id"]: p for p in corpus(5, 5)}
    for s in rp.shuffled_pairs(corpus(5, 5), seed=1):
        assert s["chosen"] == original[s["paper_id"]]["chosen"]
        assert s["prompt"] == original[s["paper_id"]]["prompt"]


def test_shuffled_is_deterministic_and_degenerate_input_is_safe():
    a = rp.shuffled_pairs(corpus(5, 5), seed=3)
    b = rp.shuffled_pairs(corpus(5, 5), seed=3)
    assert [x["donor_paper_id"] for x in a] == [x["donor_paper_id"] for x in b]
    assert rp.shuffled_pairs([pair("only")], seed=0) == []


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_summarize_reports_every_baseline():
    s = rp.summarize(rp.balance_accept_reject(corpus(), seed=0))
    for key in ("baseline_always_negative", "baseline_always_longer",
                "baseline_random", "n_against_majority", "accept_rate"):
        assert key in s
    assert s["accept_rate"] == pytest.approx(0.5)


def test_against_majority_subset_excludes_ties():
    pairs = [
        pair("a", majority_agreed_with_decision=True),
        pair("b", majority_agreed_with_decision=False),
        pair("c", majority_agreed_with_decision=None),
    ]
    assert [p["paper_id"] for p in rp.against_majority_subset(pairs)] == ["b"]
