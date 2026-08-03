"""Fully offline end-to-end tests: no hub, no downloads, no network.

Uses the hermetic tiny model from `evolving_dpo.tiny` to run the complete
pipeline — encode -> collate -> train (LoRA policy vs adapter-off reference)
-> score. These are the tests that prove OUR code works regardless of
network conditions.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from evolving_dpo.data import make_synthetic_pairs  # noqa: E402
from evolving_dpo.evaluate import OFFLINE_RUN, run_candidate, score  # noqa: E402
from evolving_dpo.losses import dpo_loss, simpo_loss  # noqa: E402
from evolving_dpo.tiny import ByteTokenizer, tiny_policy  # noqa: E402
from evolving_dpo.trainer import TrainConfig, train_dpo  # noqa: E402


def _tiny_model(seed=0):
    model, _, _ = tiny_policy(seed=seed)
    return model


def test_full_pipeline_offline():
    model = _tiny_model(seed=0)
    tok = ByteTokenizer()
    device = torch.device("cpu")

    train_cfg = TrainConfig(
        steps=6, batch_size=2, grad_accum=2, lr=1e-3,
        max_prompt_tokens=48, max_completion_tokens=48, log_every=0,
    )
    stats = train_dpo(model, tok, make_synthetic_pairs(8), dpo_loss, train_cfg, device)
    assert stats["diverged"] is False
    assert len(stats["loss"]) == 6
    assert all(x == x for x in stats["loss"])  # no NaNs

    # First-step loss must be exactly ln 2: policy == reference before any
    # update, so delta = 0 -> -logsigmoid(0) = 0.693... (the paper anchor).
    assert abs(stats["loss"][0] - 0.6931) < 1e-3

    metrics = score(model, tok, make_synthetic_pairs(6, seed=3), OFFLINE_RUN, device)
    assert 0.0 <= metrics["pref_accuracy"] <= 1.0
    assert metrics["drift_per_token"] >= 0.0
    assert metrics["n_eval_pairs"] == 6


def test_reference_free_loss_skips_reference_pass():
    """SimPO trains without ever touching the disabled-adapter path."""
    model = _tiny_model(seed=2)
    calls = {"n": 0}
    original = model.disable_adapter

    def counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    model.disable_adapter = counting

    stats = train_dpo(
        model, ByteTokenizer(), make_synthetic_pairs(6), simpo_loss,
        TrainConfig(steps=4, batch_size=2, grad_accum=2, lr=1e-3, log_every=0,
                    beta=2.0, extra_loss_kwargs={"gamma": 1.0}),
        torch.device("cpu"),
    )
    assert stats["diverged"] is False
    assert stats["used_reference"] is False
    assert calls["n"] == 0, "reference pass should be skipped entirely"
    assert stats["chosen_shift"] == []  # not measurable without a reference


def test_displacement_diagnostic_is_recorded():
    """Reference-based training must log a per-step chosen_shift."""
    stats = train_dpo(
        _tiny_model(seed=3), ByteTokenizer(), make_synthetic_pairs(6), dpo_loss,
        TrainConfig(steps=3, batch_size=2, grad_accum=2, lr=1e-3, log_every=0),
        torch.device("cpu"),
    )
    assert stats["used_reference"] is True
    assert len(stats["chosen_shift"]) == 3
    # Step 0: policy == reference, so the shift is exactly zero.
    assert abs(stats["chosen_shift"][0]) < 1e-6


def test_score_reports_displacement_metrics():
    metrics = score(_tiny_model(seed=4), ByteTokenizer(),
                    make_synthetic_pairs(4, seed=9), OFFLINE_RUN,
                    torch.device("cpu"))
    for key in ("chosen_logp_shift", "displacement_rate", "drift_per_token"):
        assert key in metrics
    assert 0.0 <= metrics["displacement_rate"] <= 1.0
    # Untrained adapters => policy == reference => no shift, no displacement.
    assert abs(metrics["chosen_logp_shift"]) < 1e-6


def test_run_candidate_offline_for_every_registered_loss():
    """The whole fitness path, hermetically, for each objective."""
    from evolving_dpo.losses import LOSS_REGISTRY, LOSS_DEFAULTS
    import dataclasses

    for name in sorted(LOSS_REGISTRY):
        cfg = OFFLINE_RUN
        d = LOSS_DEFAULTS.get(name, {})
        if d:
            train = dataclasses.replace(
                cfg.train, beta=d.get("beta", cfg.train.beta),
                extra_loss_kwargs=d.get("extra", {}),
            )
            cfg = dataclasses.replace(cfg, train=train)
        m = run_candidate(name, cfg)
        assert m["diverged"] is False, name
        assert "combined_score" in m, name
        assert m["used_reference"] is (name != "simpo"), name


def test_evolved_style_candidate_runs():
    """A weird-but-valid candidate (the kind evolution will produce) must not
    crash the harness."""
    import torch.nn.functional as F

    def candidate(pc, pr, rc, rr, beta=0.1, **kw):
        delta = beta * ((pc - pr) - (rc - rr))
        return 0.5 * (-F.logsigmoid(delta)) + 0.5 * torch.relu(1.0 - delta) ** 2

    stats = train_dpo(
        _tiny_model(seed=1), ByteTokenizer(), make_synthetic_pairs(6), candidate,
        TrainConfig(steps=4, batch_size=2, grad_accum=2, lr=1e-3, log_every=0),
        torch.device("cpu"),
    )
    assert stats["diverged"] is False
    # No attribute on an evolved body => must default to using the reference.
    assert stats["used_reference"] is True
