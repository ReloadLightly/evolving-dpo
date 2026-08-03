"""A minimal, readable DPO-family trainer.

Deliberately ~200 lines instead of a framework: every moving part of
preference optimization is visible on one screen each. The reference model
costs no extra memory — the policy is (base model + LoRA adapters), so
running the same model with adapters disabled IS the frozen reference.

    policy  = model with LoRA enabled   (trainable)
    ref     = model with LoRA disabled  (frozen, exact same base weights)

For reference-free objectives (SimPO, ORPO...) the reference passes are
skipped entirely: 4 forward passes per step become 2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .data import collate, encode_pair
from .losses import needs_reference


def sequence_logps(model, input_ids, attention_mask, labels):
    """Sum of completion-token log-probabilities per sequence.

    Station 2 as code: log P(completion | prompt) = sum over completion
    positions of log softmax(logits)[actual next token]. Positions whose
    label is -100 (prompt, padding) contribute nothing — the loss mask.

    Returns (sum_logps, completion_lengths), both shape [batch].
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: the logits at position t are the distribution over token t+1.
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    mask = labels != -100
    labels[labels == -100] = 0  # placeholder index; masked out below
    logps = torch.log_softmax(logits.float(), dim=-1)
    token_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1), mask.sum(-1).clamp(min=1)


def batch_logps(model, batch, prefix):
    return sequence_logps(
        model,
        batch[f"{prefix}_input_ids"],
        batch[f"{prefix}_attention_mask"],
        batch[f"{prefix}_labels"],
    )


@dataclass
class TrainConfig:
    beta: float = 0.1
    lr: float = 1e-5
    steps: int = 300
    batch_size: int = 2
    grad_accum: int = 8
    max_prompt_tokens: int = 384
    max_completion_tokens: int = 384
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    log_every: int = 20
    extra_loss_kwargs: dict = field(default_factory=dict)


def train_dpo(model, tokenizer, pairs, loss_fn, cfg: TrainConfig, device=None):
    """Train the policy (LoRA adapters) with `loss_fn`; return diagnostics.

    `model` must be a PEFT model. Returns a dict with loss/margin/accuracy
    histories, likelihood-displacement diagnostics, and a `diverged` flag
    (NaN/Inf guard) — candidates that blow up get fitness 0 instead of
    crashing the evolution run.

    Diagnostics worth understanding:
      margin        the implicit reward margin (reference-based) or the
                    length-normalized score gap (reference-free).
      chosen_shift  mean (policy - reference) log-probability of the CHOSEN
                    response, per token. If this goes NEGATIVE the model is
                    getting *less* likely to say the preferred answer even
                    as the margin improves — likelihood displacement
                    (Razin et al., ICLR 2025), the failure that turned a
                    refusal-trained Llama-3-8B's refusal rate from 74% to 33%.
                    Only tracked when the loss uses a reference.
    """
    device = device or next(model.parameters()).device
    rng = random.Random(cfg.seed)
    use_ref = needs_reference(loss_fn)
    encoded = [
        encode_pair(tokenizer, ex, cfg.max_prompt_tokens, cfg.max_completion_tokens)
        for ex in pairs
    ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    def lr_lambda(step):  # linear warmup, then constant
        return min(1.0, (step + 1) / max(1, cfg.warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {"loss": [], "margin": [], "accuracy": [], "chosen_shift": []}
    order, cursor = list(range(len(encoded))), 0
    rng.shuffle(order)

    model.train()
    opt.zero_grad(set_to_none=True)
    for step in range(cfg.steps):
        # ---- sample a batch (reshuffle each epoch) -----------------------
        if cursor + cfg.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        idx = order[cursor : cursor + cfg.batch_size]
        cursor += cfg.batch_size
        batch = collate(tokenizer, [encoded[i] for i in idx], device)

        # ---- policy log-probs: adapters on, gradients flowing ------------
        pol_chosen, len_chosen = batch_logps(model, batch, "chosen")
        pol_rejected, len_rejected = batch_logps(model, batch, "rejected")

        # ---- reference log-probs: same model, adapters off, no grad ------
        # Skipped entirely for reference-free losses — this is the 2x saving.
        if use_ref:
            with torch.no_grad(), model.disable_adapter():
                ref_chosen, _ = batch_logps(model, batch, "chosen")
                ref_rejected, _ = batch_logps(model, batch, "rejected")
        else:
            ref_chosen = torch.zeros_like(pol_chosen)
            ref_rejected = torch.zeros_like(pol_rejected)

        losses = loss_fn(
            pol_chosen, pol_rejected, ref_chosen, ref_rejected,
            beta=cfg.beta,
            chosen_lengths=len_chosen, rejected_lengths=len_rejected,
            **cfg.extra_loss_kwargs,
        )
        loss = losses.mean()

        if not torch.isfinite(loss):
            return {**history, "diverged": True, "steps_done": step,
                    "used_reference": use_ref}

        (loss / cfg.grad_accum).backward()
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        # ---- diagnostics -------------------------------------------------
        with torch.no_grad():
            if use_ref:
                margin = cfg.beta * (
                    (pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)
                )
                shift = ((pol_chosen - ref_chosen) / len_chosen).mean()
                history["chosen_shift"].append(float(shift))
            else:
                margin = cfg.beta * (
                    pol_chosen / len_chosen - pol_rejected / len_rejected
                )
            history["loss"].append(float(loss))
            history["margin"].append(float(margin.mean()))
            history["accuracy"].append(float((margin > 0).float().mean()))

        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            k = cfg.log_every
            msg = (
                f"step {step + 1:>4}/{cfg.steps}  "
                f"loss {sum(history['loss'][-k:]) / k:.4f}  "
                f"margin {sum(history['margin'][-k:]) / k:+.3f}  "
                f"acc {sum(history['accuracy'][-k:]) / k:.3f}"
            )
            if use_ref:
                msg += f"  chosen_shift {sum(history['chosen_shift'][-k:]) / k:+.4f}"
            print(msg)

    return {**history, "diverged": False, "steps_done": cfg.steps,
            "used_reference": use_ref}
