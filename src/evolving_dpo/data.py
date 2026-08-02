"""Preference-pair loading, tokenization, and batching.

Default dataset: HuggingFaceH4/ultrafeedback_binarized
  * splits: train_prefs / test_prefs
  * fields: "prompt" (str), "chosen" / "rejected" (chat transcripts —
    lists of {"role": ..., "content": ...}); the completion we train on is
    the final assistant turn.

Lecture-0 sightings in this file (the masks, in the flesh):
  * loss mask   -> labels are -100 on prompt and padding tokens, so
                   cross-entropy/log-probs only count completion tokens.
  * padding mask-> attention_mask is 0 on pads.
  * causal mask -> lives inside the model; you never see it here.
"""

from __future__ import annotations

import random
from typing import Any

import torch

DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


def _last_assistant(messages: Any) -> str:
    """Extract the completion text from a chat-transcript field."""
    if isinstance(messages, str):
        return messages
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    # Fallback: last message content, whatever its role.
    last = messages[-1]
    return last.get("content", "") if isinstance(last, dict) else str(last)


def load_pairs(
    n_train: int,
    n_eval: int,
    dataset_name: str = DEFAULT_DATASET,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Return (train_pairs, eval_pairs) as lists of
    {"prompt": str, "chosen": str, "rejected": str}."""
    from datasets import load_dataset  # lazy: keeps CPU smoke tests light

    train_ds = load_dataset(dataset_name, split="train_prefs")
    eval_ds = load_dataset(dataset_name, split="test_prefs")
    train_ds = train_ds.shuffle(seed=seed).select(range(min(n_train, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=seed).select(range(min(n_eval, len(eval_ds))))

    def simplify(ds) -> list[dict]:
        out = []
        for ex in ds:
            chosen = _last_assistant(ex["chosen"])
            rejected = _last_assistant(ex["rejected"])
            if not chosen or not rejected or chosen == rejected:
                continue  # degenerate pair: no preference signal
            out.append(
                {"prompt": ex["prompt"], "chosen": chosen, "rejected": rejected}
            )
        return out

    return simplify(train_ds), simplify(eval_ds)


def make_synthetic_pairs(n: int = 16, seed: int = 0) -> list[dict]:
    """Tiny handwritten pairs for offline smoke tests (no downloads)."""
    rng = random.Random(seed)
    topics = ["the moon", "sorting lists", "green tea", "Bangkok traffic",
              "entropy", "house cats", "volcanoes", "chess openings"]
    pairs = []
    for i in range(n):
        t = topics[i % len(topics)]
        pairs.append(
            {
                "prompt": f"Explain {t} in one sentence.",
                "chosen": f"{t.capitalize()} can be summarized clearly: "
                          f"it is best understood through one key idea, stated simply.",
                "rejected": rng.choice(["idk", "no.", "askjdh", "It is what it is."]),
            }
        )
    return pairs


def render_prompt(tokenizer, prompt: str) -> str:
    """Format the user prompt for the model.

    Uses the tokenizer's chat template when available (Qwen etc.); falls back
    to a plain format for template-less tokenizers (e.g. GPT-2 in tests).
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            # Qwen3 templates accept enable_thinking; disable reasoning mode
            # for plain preference tuning.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return f"User: {prompt}\nAssistant: "


def encode_pair(
    tokenizer,
    ex: dict,
    max_prompt_tokens: int = 384,
    max_completion_tokens: int = 384,
) -> dict:
    """Tokenize one preference pair into (ids, labels) for chosen & rejected.

    labels: -100 on every prompt token (no loss there), the token id itself on
    completion tokens. EOS is appended so the model also learns to stop.
    """
    prompt_ids = tokenizer(render_prompt(tokenizer, ex["prompt"]),
                           add_special_tokens=False).input_ids
    # Keep the END of an over-long prompt: it is the context nearest the answer.
    prompt_ids = prompt_ids[-max_prompt_tokens:]

    out = {}
    for key in ("chosen", "rejected"):
        comp_ids = tokenizer(ex[key], add_special_tokens=False).input_ids
        comp_ids = comp_ids[:max_completion_tokens]
        if tokenizer.eos_token_id is not None:
            comp_ids = comp_ids + [tokenizer.eos_token_id]
        ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + list(comp_ids)
        out[key] = {"input_ids": ids, "labels": labels}
    return out


def collate(tokenizer, encoded: list[dict], device: torch.device) -> dict:
    """Pad a list of encoded pairs into batch tensors.

    Returns a dict with chosen_/rejected_ x input_ids/attention_mask/labels.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch = {}
    for key in ("chosen", "rejected"):
        seqs = [e[key] for e in encoded]
        max_len = max(len(s["input_ids"]) for s in seqs)
        input_ids, attention, labels = [], [], []
        for s in seqs:
            n_pad = max_len - len(s["input_ids"])
            input_ids.append(s["input_ids"] + [pad_id] * n_pad)
            attention.append([1] * len(s["input_ids"]) + [0] * n_pad)
            labels.append(s["labels"] + [-100] * n_pad)
        batch[f"{key}_input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
        batch[f"{key}_attention_mask"] = torch.tensor(attention, dtype=torch.long, device=device)
        batch[f"{key}_labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return batch
