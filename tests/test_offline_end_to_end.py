"""Fully offline end-to-end test: no hub, no downloads.

Builds a tiny GPT-2 from a config and a byte-level fake tokenizer, then runs
the complete pipeline — encode → collate → train (LoRA policy vs adapter-off
reference) → score. This is the test that proves OUR code works regardless of
network conditions.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from evolving_dpo.data import make_synthetic_pairs  # noqa: E402
from evolving_dpo.evaluate import DRY_RUN, score  # noqa: E402
from evolving_dpo.losses import dpo_loss  # noqa: E402
from evolving_dpo.trainer import TrainConfig, train_dpo  # noqa: E402


class ByteTokenizer:
    """Minimal offline tokenizer: bytes + 2 -> ids (0=pad, 1=eos)."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = None

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Enc([b + 2 for b in text.encode("utf-8")])


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    cfg = GPT2Config(
        vocab_size=258, n_positions=256, n_embd=32, n_layer=2, n_head=2,
        # Zero dropout so policy == reference exactly at init (the ln 2 anchor
        # below is exact). Qwen-class models have no dropout anyway.
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=1, eos_token_id=1,
    )
    model = GPT2LMHeadModel(cfg).float()
    lora = LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora)


def test_full_pipeline_offline():
    torch.manual_seed(0)
    model = _tiny_model()
    tok = ByteTokenizer()
    device = torch.device("cpu")

    train_cfg = TrainConfig(
        steps=6, batch_size=2, grad_accum=2, lr=1e-3,
        max_prompt_tokens=48, max_completion_tokens=48, log_every=0,
    )
    pairs = make_synthetic_pairs(8)

    stats = train_dpo(model, tok, pairs, dpo_loss, train_cfg, device)
    assert stats["diverged"] is False
    assert len(stats["loss"]) == 6
    assert all(map(lambda x: x == x, stats["loss"]))  # no NaNs

    # First-step loss must be exactly ln 2: policy == reference before any
    # update, so delta = 0 -> -logsigmoid(0) = 0.693... (the paper anchor).
    assert abs(stats["loss"][0] - 0.6931) < 1e-3

    metrics = score(model, tok, make_synthetic_pairs(6, seed=3), DRY_RUN, device)
    assert 0.0 <= metrics["pref_accuracy"] <= 1.0
    assert metrics["drift_per_token"] >= 0.0
    assert metrics["n_eval_pairs"] == 6


def test_evolved_style_candidate_runs():
    """A weird-but-valid candidate (the kind evolution will produce) must not
    crash the harness."""
    import torch.nn.functional as F

    def candidate(pc, pr, rc, rr, beta=0.1, **kw):
        delta = beta * ((pc - pr) - (rc - rr))
        return 0.5 * (-F.logsigmoid(delta)) + 0.5 * torch.relu(1.0 - delta) ** 2

    torch.manual_seed(1)
    model = _tiny_model()
    stats = train_dpo(
        model, ByteTokenizer(), make_synthetic_pairs(6), candidate,
        TrainConfig(steps=4, batch_size=2, grad_accum=2, lr=1e-3, log_every=0),
        torch.device("cpu"),
    )
    assert stats["diverged"] is False
