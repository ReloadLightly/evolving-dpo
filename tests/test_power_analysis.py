"""Offline tests for the power analysis (scripts/power_analysis.py).

Power formulas fail quietly: a wrong constant still produces a plausible
number, and the mistake only surfaces as an underpowered study months later.
So these check the arithmetic against a closed form and against the
qualitative relations that must hold for any correct implementation.
"""

import importlib.util
import math
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "power_analysis.py"
_spec = importlib.util.spec_from_file_location("power_analysis", _SCRIPT)
pa = importlib.util.module_from_spec(_spec)
sys.modules["power_analysis"] = pa
_spec.loader.exec_module(pa)


def test_unpaired_matches_the_closed_form():
    """n = (z_a*sqrt(2*pq) + z_b*sqrt(p1q1+p2q2))^2 / d^2, at p = 0.5."""
    expected = (1.959963985 * math.sqrt(2 * 0.25)
                + 0.841621234 * math.sqrt(0.5)) / math.sqrt(1000)
    assert pa.mde_unpaired(1000, 0.5) == pytest.approx(expected, abs=5e-4)


def test_more_data_detects_smaller_effects():
    sizes = [250, 500, 1000, 2000, 4000]
    mdes = [pa.mde_unpaired(n, 0.6) for n in sizes]
    assert mdes == sorted(mdes, reverse=True)


def test_mde_scales_as_one_over_sqrt_n():
    """Quadrupling the sample should roughly halve the detectable effect."""
    assert pa.mde_unpaired(4000, 0.5) == pytest.approx(
        pa.mde_unpaired(1000, 0.5) / 2, rel=0.05)


def test_paired_beats_unpaired():
    """Scoring both arms on the same pairs is strictly more informative."""
    for n in (500, 1000, 2000):
        assert pa.mde_paired(n, 0.25) < pa.mde_unpaired(n, 0.5)


def test_paired_power_depends_on_discordance():
    """More disagreement between arms means a larger detectable difference."""
    assert pa.mde_paired(1000, 0.15) < pa.mde_paired(1000, 0.35)


def test_interaction_is_harder_than_a_main_effect():
    """Four cells' variance, so ~2x the SE and a larger detectable effect."""
    for n in (500, 1000, 2000):
        assert pa.mde_interaction(n, 0.6) > pa.mde_unpaired(n, 0.6)


def test_n_for_interaction_inverts_mde_interaction():
    for target in (0.03, 0.05, 0.10):
        n = pa.n_for_interaction(target, 0.6)
        assert pa.mde_interaction(n, 0.6) <= target
        # And one grid step smaller must not have sufficed.
        assert pa.mde_interaction(n - 50, 0.6) > target


def test_non_overlapping_ci_is_stricter_than_a_t_test():
    """The preregistration's criterion is conservative; say so with a number."""
    m = pa.mde_seed_level(0.01, n_seeds=5)
    assert m["non_overlapping_ci"] > m["two_sample_t"]
    assert 1.1 < m["ratio"] < 1.4


def test_seed_level_mde_is_linear_in_seed_sd():
    a = pa.mde_seed_level(0.01)["non_overlapping_ci"]
    b = pa.mde_seed_level(0.02)["non_overlapping_ci"]
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_balanced_eval_is_twice_the_minority_class():
    """The 50/50 requirement is bound by the smaller class (HANDOFF s6)."""
    assert pa.balanced_eval_size(3488, 1101) == 2202
    assert pa.balanced_eval_size(1000, 500) == 1000     # already balanced
    assert pa.balanced_eval_size(1000, 900) == 200      # badly skewed


def test_projected_counts_are_internally_consistent():
    for year, d in pa.PROJECTED.items():
        assert 0 < d["accepted"] < d["pairs"], year
        assert 0 <= d["against_majority"] <= d["pairs"], year


def test_report_is_written(tmp_path):
    out = tmp_path / "power.md"
    sys.argv = ["power_analysis.py", "--out", str(out)]
    assert pa.main() == 0
    body = out.read_text(encoding="utf-8")
    assert "H3" in body and "interaction" in body
    assert "balanc" in body.lower()
