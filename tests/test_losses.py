"""Unit tests for the loss plug-ins — including the pen-and-paper anchors."""

import math
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from evolving_dpo.losses import dpo_loss, ipo_loss, slic_loss, dpop_loss, LOSS_REGISTRY  # noqa: E402


def _tensors(pc, pr, rc, rr):
    return tuple(torch.tensor([v], dtype=torch.float32) for v in (pc, pr, rc, rr))


def test_dpo_at_zero_margin_is_ln2():
    # delta = 0  ->  -log sigmoid(0) = ln 2: one bit of surprise, on paper and here.
    loss = dpo_loss(*_tensors(-10.0, -10.0, -3.0, -3.0), beta=0.1)
    assert abs(float(loss) - math.log(2)) < 1e-5


def test_dpo_monotone_in_margin():
    lo = dpo_loss(*_tensors(-9.0, -10.0, -3.0, -3.0), beta=0.1)   # small margin
    hi = dpo_loss(*_tensors(-5.0, -10.0, -3.0, -3.0), beta=0.1)   # big margin
    assert float(hi) < float(lo)


def test_ipo_minimum_at_target_margin():
    beta = 0.1
    target = 1.0 / (2 * beta)  # = 5.0 in log-ratio units
    at_target = ipo_loss(*_tensors(-5.0 + target, -5.0, -5.0, -5.0), beta=beta)
    off_target = ipo_loss(*_tensors(-5.0 + target + 1.0, -5.0, -5.0, -5.0), beta=beta)
    assert float(at_target) < 1e-6
    assert float(off_target) > float(at_target)


def test_slic_zero_beyond_margin():
    # delta = beta * 20 = 2.0 >= 1 -> hinge closed.
    loss = slic_loss(*_tensors(-1.0, -21.0, -3.0, -3.0), beta=0.1)
    assert float(loss) == 0.0


def test_dpop_penalizes_chosen_likelihood_drop():
    args_ok = _tensors(-3.0, -10.0, -4.0, -4.0)     # policy_chosen ABOVE ref_chosen
    args_bad = _tensors(-6.0, -13.0, -4.0, -4.0)    # same margin, chosen degraded
    base_ok, base_bad = dpo_loss(*args_ok, beta=0.1), dpo_loss(*args_bad, beta=0.1)
    assert abs(float(base_ok) - float(base_bad)) < 1e-6  # DPO can't tell them apart...
    ok, bad = dpop_loss(*args_ok, beta=0.1), dpop_loss(*args_bad, beta=0.1)
    assert float(bad) > float(ok)                        # ...DPOP can.


def test_registry_signatures():
    args = _tensors(-4.0, -9.0, -5.0, -5.0)
    for name, fn in LOSS_REGISTRY.items():
        out = fn(*args, beta=0.1)
        assert out.shape == (1,), name
        assert torch.isfinite(out).all(), name
