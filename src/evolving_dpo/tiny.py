"""A hermetic tiny model + tokenizer for plumbing checks.

Everything here is constructed locally — no hub, no downloads, no network.
Use it to prove the pipeline runs end to end (encode -> collate -> train ->
score) before spending GPU minutes on a real model, and to keep CI honest.

The model is a 2-layer GPT-2 with a 258-token byte vocabulary: nonsense
outputs, correct plumbing.
"""

from __future__ import annotations

import torch


class ByteTokenizer:
    """Minimal offline tokenizer: UTF-8 bytes + 2 -> ids (0=pad, 1=eos)."""

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


def tiny_policy(lora_r: int = 4, seed: int | None = None):
    """Return (peft_model, tokenizer, device) with the same contract as
    evaluate.load_policy — but built from a config, so it never touches the
    network."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    if seed is not None:
        torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=258, n_positions=256, n_embd=32, n_layer=2, n_head=2,
        # Zero dropout so policy == reference exactly at initialization:
        # the first DPO step then costs exactly ln 2, which is a checkable
        # invariant rather than a vibe.
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=1, eos_token_id=1,
    )
    model = GPT2LMHeadModel(cfg).float()
    lora = LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), ByteTokenizer(), torch.device("cpu")
