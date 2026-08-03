"""Candidate fitness: train briefly with a loss function, score on held-out pairs.

This one function is used two ways:
  * Phases 0–2: called manually (scripts/run_baseline.py) to benchmark
    known losses — DPO, IPO, SLiC, DPOP, SimPO.
  * Phase 3: called by ShinkaEvolve (shinka_task/) thousands of times with
    evolved loss functions.

Fitness v0 — simple on purpose, guarded on purpose:

    combined_score = pref_accuracy
                     - drift_penalty * max(0, drift - drift_budget)
                     - displacement_penalty * max(0, -chosen_logp_shift)

  pref_accuracy       fraction of held-out pairs where the implicit reward
                      margin is positive (the trained policy ranks chosen
                      above rejected).
  drift               mean per-token |policy logp - ref logp| on held-out
                      CHOSEN completions — a cheap KL-style proxy (Station 4)
                      for "how far has the policy wandered".
  chosen_logp_shift   SIGNED mean per-token (policy - ref) on held-out chosen
                      completions. Negative = likelihood displacement: the
                      preferred answer became LESS likely while the margin
                      improved. `displacement_penalty` defaults to 0.0, so
                      this is measured but not optimized against — measure
                      before you optimize. Raise it deliberately, and record
                      that you did, because it changes what fitness means.
  diverged            NaN/Inf during training -> score 0.

Evaluation ALWAYS computes reference log-probabilities, even for
reference-free losses like SimPO. Training may skip them for speed, but
metrics must stay comparable across the whole registry.

The drift term is the Goodhart guard: without it, evolution's first
discovery is usually a loss that wins preference accuracy by destroying the
model. Expect candidates to probe this guard's edges. That is a finding,
not a bug — document it when it happens.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field, asdict

import torch

from .data import load_pairs, make_synthetic_pairs, collate, encode_pair
from .losses import get_loss
from .trainer import TrainConfig, train_dpo, batch_logps


@dataclass
class EvalConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    n_train: int = 2000
    n_eval: int = 500
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    synthetic_data: bool = False        # handwritten pairs, no dataset download
    offline_model: bool = False         # build a tiny model locally, no hub
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_batch_size: int = 4
    drift_penalty: float = 2.0
    drift_budget: float = 0.15          # nats/token of free movement
    displacement_penalty: float = 0.0   # off by default — diagnostic first
    seed: int = 0
    device: str | None = None           # None -> cuda if available


#: Plumbing check against a real (tiny) hub model — needs internet once.
DRY_RUN = EvalConfig(
    model_id="sshleifer/tiny-gpt2",
    n_train=16,
    n_eval=8,
    synthetic_data=True,
    train=TrainConfig(steps=4, batch_size=2, grad_accum=2,
                      max_prompt_tokens=48, max_completion_tokens=48,
                      log_every=2, lr=1e-3),
    eval_batch_size=4,
)

#: Same, but hermetic: model built from a config, tokenizer is bytes.
#: No network at all. This is the one to run when you just want to know
#: whether the code works.
OFFLINE_RUN = dataclasses.replace(
    DRY_RUN, model_id="<offline-tiny-gpt2>", offline_model=True
)


def _lora_targets(model_id: str):
    # GPT-2-family (used in dry runs/tests) needs explicit Conv1D targets.
    if "gpt2" in model_id.lower():
        return ["c_attn"]
    return "all-linear"


def load_policy(cfg: EvalConfig):
    """Base model + LoRA adapters = policy; adapters disabled = reference."""
    if cfg.offline_model:
        from .tiny import tiny_policy
        return tiny_policy(seed=cfg.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # .float() keeps things version-proof (dtype kwargs changed across
    # transformers majors) and T4-safe (no accidental bf16 on old GPUs).
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id).float().to(device)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=_lora_targets(cfg.model_id),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tokenizer, device


@torch.no_grad()
def score(model, tokenizer, eval_pairs, cfg: EvalConfig, device) -> dict:
    """Held-out metrics for a trained policy.

    Always reference-based, so numbers are comparable across every loss in
    the registry regardless of what the loss itself needed during training.
    """
    model.eval()
    encoded = [
        encode_pair(tokenizer, ex,
                    cfg.train.max_prompt_tokens, cfg.train.max_completion_tokens)
        for ex in eval_pairs
    ]
    margins, drifts, shifts = [], [], []
    bs = cfg.eval_batch_size
    for i in range(0, len(encoded), bs):
        batch = collate(tokenizer, encoded[i : i + bs], device)
        with model.disable_adapter():
            ref_c, len_c = batch_logps(model, batch, "chosen")
            ref_r, _ = batch_logps(model, batch, "rejected")
        pol_c, _ = batch_logps(model, batch, "chosen")
        pol_r, _ = batch_logps(model, batch, "rejected")
        margin = cfg.train.beta * ((pol_c - pol_r) - (ref_c - ref_r))
        signed = (pol_c - ref_c) / len_c
        margins += margin.tolist()
        drifts += signed.abs().tolist()
        shifts += signed.tolist()

    n = max(1, len(margins))
    return {
        "pref_accuracy": sum(m > 0 for m in margins) / n,
        "mean_margin": sum(margins) / n,
        "drift_per_token": sum(drifts) / max(1, len(drifts)),
        "chosen_logp_shift": sum(shifts) / max(1, len(shifts)),
        "displacement_rate": sum(s < 0 for s in shifts) / max(1, len(shifts)),
        "n_eval_pairs": len(margins),
    }


def run_candidate(loss_fn_or_name, cfg: EvalConfig | None = None) -> dict:
    """Train + evaluate one candidate loss. Returns a flat metrics dict
    with `combined_score` (higher is better) — the contract ShinkaEvolve sees.
    """
    cfg = cfg or EvalConfig()
    loss_fn = (
        get_loss(loss_fn_or_name)
        if isinstance(loss_fn_or_name, str)
        else loss_fn_or_name
    )
    torch.manual_seed(cfg.seed)

    t0 = time.time()
    model, tokenizer, device = load_policy(cfg)

    if cfg.synthetic_data:
        train_pairs = make_synthetic_pairs(cfg.n_train, seed=cfg.seed)
        eval_pairs = make_synthetic_pairs(cfg.n_eval, seed=cfg.seed + 1)
    else:
        train_pairs, eval_pairs = load_pairs(
            cfg.n_train, cfg.n_eval, cfg.dataset, seed=cfg.seed
        )

    try:
        stats = train_dpo(model, tokenizer, train_pairs, loss_fn, cfg.train, device)
    except RuntimeError as e:  # OOM or numerical explosion inside a candidate
        return {
            "combined_score": 0.0, "diverged": True, "error": str(e)[:500],
            "seconds": time.time() - t0,
        }

    if stats["diverged"]:
        return {
            "combined_score": 0.0, "diverged": True,
            "steps_done": stats["steps_done"], "seconds": time.time() - t0,
        }

    metrics = score(model, tokenizer, eval_pairs, cfg, device)
    over_budget = max(0.0, metrics["drift_per_token"] - cfg.drift_budget)
    displaced = max(0.0, -metrics["chosen_logp_shift"])
    metrics.update(
        combined_score=(
            metrics["pref_accuracy"]
            - cfg.drift_penalty * over_budget
            - cfg.displacement_penalty * displaced
        ),
        diverged=False,
        used_reference=stats["used_reference"],
        final_train_loss=stats["loss"][-1] if stats["loss"] else None,
        seconds=time.time() - t0,
        config={"model_id": cfg.model_id, "n_train": cfg.n_train,
                "n_eval": cfg.n_eval,
                "displacement_penalty": cfg.displacement_penalty,
                **asdict(cfg.train)},
    )
    return metrics
