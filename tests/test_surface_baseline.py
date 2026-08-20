"""Tests for the surface-feature classifier (evolving_dpo.surface_baseline).

This baseline decides H2, so it has to be strong enough to be a fair test. Two
failure modes would make it useless in opposite directions:

  * too weak -- a broken optimizer that cannot learn a signal that is really
    there would let the language model 'beat style' trivially;
  * secretly perfect -- if the pair ordering were not randomized, every label
    would be 1 and the classifier would score 100% while measuring nothing.

Both are tested directly.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from evolving_dpo import surface_baseline as sb  # noqa: E402


def pair(chosen, rejected, **kw):
    p = {"prompt": "T\n\nA", "chosen": chosen, "rejected": rejected}
    p.update(kw)
    return p


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_feature_vector_shape_matches_names():
    assert sb.features("A review.").shape == (len(sb.FEATURE_NAMES),)


def test_features_count_what_they_claim():
    text = ("The method may be unclear. Is it correct? What about [1] and "
            "(Smith et al., 2020)? We tried 3 and 4.5 values.")
    f = dict(zip(sb.FEATURE_NAMES, sb.features(text)))
    assert f["n_questions"] == 2
    assert f["n_hedges"] >= 2            # 'may', 'unclear'
    assert f["n_citations"] == 2         # [1] and (Smith et al., 2020)
    assert f["n_numbers"] >= 2           # 3 and 4.5
    assert f["n_words"] > 0


def test_sentiment_polarity_is_signed_and_bounded():
    pos = dict(zip(sb.FEATURE_NAMES, sb.features("novel strong clear elegant")))
    neg = dict(zip(sb.FEATURE_NAMES, sb.features("weak flawed confusing poor")))
    assert pos["sentiment"] == pytest.approx(1.0)
    assert neg["sentiment"] == pytest.approx(-1.0)
    assert -1.0 <= dict(zip(sb.FEATURE_NAMES, sb.features("novel weak")))["sentiment"] <= 1.0


def test_empty_text_does_not_divide_by_zero():
    f = sb.features("")
    assert np.all(np.isfinite(f))


def test_section_headings_are_counted():
    f = dict(zip(sb.FEATURE_NAMES, sb.features("## Summary\n\nx\n\n## Weaknesses\n\ny")))
    assert f["n_sections"] == 2


# --------------------------------------------------------------------------
# The design matrix -- the subtle part
# --------------------------------------------------------------------------

def test_labels_are_balanced_not_constant():
    """If ordering were not randomized every label would be 1 and the
    classifier would score 100% while learning nothing."""
    pairs = [pair(f"chosen text {i} " * 10, f"rejected {i} " * 5) for i in range(200)]
    X, y = sb.build_design(pairs, seed=0)
    assert len(X) == 200
    assert 0.3 < y.mean() < 0.7, "presentation order must be randomized"


def test_design_is_deterministic_for_a_seed():
    pairs = [pair(f"a {i} " * 8, f"b {i} " * 4) for i in range(50)]
    X1, y1 = sb.build_design(pairs, seed=5)
    X2, y2 = sb.build_design(pairs, seed=5)
    assert np.allclose(X1, X2) and np.array_equal(y1, y2)


def test_rows_are_antisymmetric():
    """Flipping the presentation order must negate the features."""
    p = [pair("novel strong thorough " * 10, "weak flawed " * 3)]
    X_pos, y_pos = sb.build_design(p, seed=0)
    for seed in range(30):
        X, y = sb.build_design(p, seed=seed)
        if y[0] != y_pos[0]:
            assert np.allclose(X[0], -X_pos[0])
            return
    pytest.skip("no opposite ordering drawn in 30 seeds")


def test_pairs_missing_text_are_skipped():
    X, y = sb.build_design([pair("", "something"), pair("a b c", "")], seed=0)
    assert len(X) == 0


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------

def test_learns_a_real_length_signal():
    """Chosen always much longer: a competent baseline must find this."""
    pairs = [pair("word " * 200, "word " * 20) for _ in range(300)]
    X, y = sb.build_design(pairs, seed=0)
    clf = sb.SurfaceClassifier().fit(X, y)
    assert clf.accuracy(X, y) > 0.95


def test_cannot_learn_from_identical_features():
    """Two indistinguishable reviews must leave it at chance."""
    pairs = [pair("word " * 50, "word " * 50) for _ in range(200)]
    X, y = sb.build_design(pairs, seed=0)
    clf = sb.SurfaceClassifier().fit(X, y)
    assert 0.35 < clf.accuracy(X, y) < 0.65


def test_weights_name_the_dominant_feature():
    """Reading the weights must reveal WHICH surface cue carries the signal."""
    pairs = [pair("word " * 200, "word " * 20) for _ in range(300)]
    X, y = sb.build_design(pairs, seed=0)
    w = sb.SurfaceClassifier().fit(X, y).weights()
    assert list(w)[0] in ("n_words", "n_sentences"), f"expected length to dominate, got {list(w)[:3]}"


def test_standardization_survives_a_constant_feature():
    pairs = [pair("word " * (10 + i), "word " * 5) for i in range(60)]
    X, y = sb.build_design(pairs, seed=0)
    X[:, sb.FEATURE_NAMES.index("n_citations")] = 0.0   # constant column
    clf = sb.SurfaceClassifier().fit(X, y)
    assert np.all(np.isfinite(clf.w))


def test_fitting_nothing_raises():
    with pytest.raises(ValueError):
        sb.SurfaceClassifier().fit(np.zeros((0, len(sb.FEATURE_NAMES))), np.zeros(0))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_evaluate_surface_reports_everything_a_table_needs():
    train = [pair("novel thorough " * 60, "weak " * 10) for _ in range(200)]
    test = [pair("novel thorough " * 60, "weak " * 10) for _ in range(80)]
    out = sb.evaluate_surface(train, test, seed=0)
    for key in ("accuracy", "train_accuracy", "n_train", "n_test", "weights"):
        assert key in out
    assert out["accuracy"] > 0.9
    assert out["n_train"] == 200 and out["n_test"] == 80


def test_evaluate_surface_handles_empty_input():
    out = sb.evaluate_surface([], [], seed=0)
    assert np.isnan(out["accuracy"])


def test_generalizes_rather_than_memorizes():
    """Train and test drawn from the same rule: test accuracy must hold up."""
    rng = np.random.default_rng(0)
    def make(n):
        out = []
        for _ in range(n):
            long_chosen = rng.random() < 0.8
            c = "word " * (150 if long_chosen else 30)
            r = "word " * (30 if long_chosen else 150)
            out.append(pair(c, r))
        return out
    out = sb.evaluate_surface(make(400), make(200), seed=0)
    assert out["accuracy"] > 0.7
    assert abs(out["train_accuracy"] - out["accuracy"]) < 0.15
