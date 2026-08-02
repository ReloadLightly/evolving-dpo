"""ShinkaEvolve seed program (Phase 3).

Only the code between the EVOLVE-BLOCK markers is rewritten by the LLM
mutation engine; everything else is fixed harness. The seed is vanilla DPO —
so generation 0 of the evolution is exactly our Phase-0 baseline, and every
improvement is measured against it.
"""

# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F


def preference_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
):
    """Candidate preference-optimization objective.

    Inputs are 1-D tensors (one entry per preference pair) of summed
    completion log-probabilities. Must return a 1-D tensor of per-example
    losses (lower = better). Keep it differentiable.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    delta = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(delta)
# EVOLVE-BLOCK-END


def run_experiment(**kwargs) -> dict:
    """Entry point ShinkaEvolve calls. Trains a small model with the candidate
    loss above and returns the metrics dict (incl. `combined_score`)."""
    from evolving_dpo.evaluate import EvalConfig, run_candidate
    from evolving_dpo.trainer import TrainConfig

    cfg = EvalConfig(
        model_id=kwargs.get("model_id", "Qwen/Qwen3-0.6B"),
        n_train=int(kwargs.get("n_train", 1000)),
        n_eval=int(kwargs.get("n_eval", 300)),
        train=TrainConfig(
            steps=int(kwargs.get("steps", 200)),
            beta=float(kwargs.get("beta", 0.1)),
        ),
        seed=int(kwargs.get("seed", 0)),
    )
    return run_candidate(preference_loss, cfg)


if __name__ == "__main__":
    # Manual check of the seed program (uses the tiny dry-run config).
    from evolving_dpo.evaluate import DRY_RUN, run_candidate

    print(run_candidate(preference_loss, DRY_RUN))
