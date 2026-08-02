"""End-to-end smoke tests on CPU with a ~1 MB model (sshleifer/tiny-gpt2).

Verifies the Lecture-0 masks are wired correctly and that a few training
steps run, produce finite numbers, and actually move the adapter weights.
Requires network access to huggingface.co the first time (tiny download);
skips gracefully offline.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from evolving_dpo.data import make_synthetic_pairs, encode_pair, collate  # noqa: E402
from evolving_dpo.evaluate import DRY_RUN, run_candidate, load_policy  # noqa: E402


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as e:  # offline or hub hiccup
        pytest.skip(f"tokenizer unavailable: {e}")


def test_masks_and_collation():
    tok = _tokenizer()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ex = {"prompt": "Say hi.", "chosen": "Hello there, friend!", "rejected": "no"}
    enc = encode_pair(tok, ex, max_prompt_tokens=32, max_completion_tokens=32)

    for key in ("chosen", "rejected"):
        ids, labels = enc[key]["input_ids"], enc[key]["labels"]
        assert len(ids) == len(labels)
        n_prompt = sum(1 for l in labels if l == -100)
        assert n_prompt > 0                       # prompt tokens are loss-masked
        tail_ids, tail_labels = ids[n_prompt:], labels[n_prompt:]
        assert tail_ids == tail_labels            # completion tokens carry loss
        assert tail_labels[-1] == tok.eos_token_id  # EOS is trained

    batch = collate(tok, [enc, enc], torch.device("cpu"))
    att, lab = batch["chosen_attention_mask"], batch["chosen_labels"]
    assert ((att == 0) <= (lab == -100)).all()    # padded positions never carry loss


def test_dry_run_end_to_end():
    try:
        m = run_candidate("dpo", DRY_RUN)
    except OSError as e:
        pytest.skip(f"hub unavailable: {e}")
    assert "combined_score" in m
    assert m["diverged"] is False
    assert 0.0 <= m["pref_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(m["mean_margin"]))


def test_adapters_actually_train():
    try:
        model, tok, device = load_policy(DRY_RUN)
    except OSError as e:
        pytest.skip(f"hub unavailable: {e}")
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters() if p.requires_grad
    }
    assert before, "LoRA attached no trainable parameters"

    from evolving_dpo.trainer import train_dpo
    from evolving_dpo.losses import dpo_loss

    pairs = make_synthetic_pairs(8)
    stats = train_dpo(model, tok, pairs, dpo_loss, DRY_RUN.train, device)
    assert stats["diverged"] is False
    moved = any(
        not torch.equal(before[n], p.detach())
        for n, p in model.named_parameters() if p.requires_grad
    )
    assert moved, "no adapter weight changed after training steps"
