"""Preference-optimization loss functions — the plug-in slot this project evolves.

Every loss consumes the same four log-probabilities and returns a per-example
loss tensor. This uniform signature is what makes the objective *evolvable*:
to ShinkaEvolve, a loss is just a short function body to rewrite.

Notation (each argument is a 1-D tensor, one entry per preference pair):

    policy_chosen_logps    log pi_theta(y_chosen  | x)   summed over completion tokens
    policy_rejected_logps  log pi_theta(y_rejected| x)
    ref_chosen_logps       log pi_ref (y_chosen  | x)
    ref_rejected_logps     log pi_ref (y_rejected| x)

The central quantity is

    delta = beta * [(policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)]

i.e. beta times "how much MORE the policy prefers chosen over rejected,
relative to the reference model". Equivalently, with DPO's implicit reward
r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x)), delta is the implicit
reward margin r(chosen) - r(rejected).

Pen-and-paper anchors:
  * log-probabilities and their differences: Station 2 (chain rule, log space).
  * -logsigmoid(delta) at delta = 0 equals ln 2 ~= 0.693 — exactly one bit of
    surprise: the model is maximally unsure which answer is better (Station 3).
  * the (policy - reference) log-ratio is the same object KL regularization
    is made of (Station 4): DPO's beta is the strength of an implicit KL leash.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Original DPO (Rafailov et al., 2023).

    Bradley–Terry maximum likelihood on the implicit reward margin:
    loss = -log sigmoid(delta). Monotone: any increase of the margin helps,
    forever — which is exactly why DPO can over-optimize (see IPO).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    delta = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(delta)


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """IPO (Azar et al., 2023).

    Squared distance to a FIXED target margin 1/(2*beta): once the margin is
    reached, the gradient vanishes. A bounded objective — the "anti-Goodhart"
    response to DPO's monotone appetite.
    """
    logits = (policy_chosen_logps - policy_rejected_logps) - (
        ref_chosen_logps - ref_rejected_logps
    )
    return (logits - 1.0 / (2.0 * beta)) ** 2


def slic_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """SLiC-style hinge (Zhao et al., 2023).

    max(0, 1 - delta): linear pressure until the margin reaches 1, then done.
    The SVM move, transplanted to preference tuning.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    delta = beta * (pi_logratios - ref_logratios)
    return torch.relu(1.0 - delta)


def dpop_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    lam: float = 5.0,
    **kwargs,
) -> torch.Tensor:
    """DPO-Positive (Pal et al., 2024).

    DPO plus a penalty whenever the policy's likelihood of the CHOSEN answer
    drops below the reference's. Counteracts DPO's known failure mode of
    winning the margin by making BOTH answers less likely (margin up,
    absolute quality down — a mini reward hack).
    """
    base = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    penalty = torch.relu(ref_chosen_logps - policy_chosen_logps)
    return base + lam * penalty


# Phase 1 exercise (deliberately not implemented here): add DiscoPOP's
# discovered LRML loss by reading it out of https://github.com/SakanaAI/DiscoPOP
# and registering it below. If our evolution later rediscovers something
# LRML-shaped, that's a result.

LOSS_REGISTRY = {
    "dpo": dpo_loss,
    "ipo": ipo_loss,
    "slic": slic_loss,
    "dpop": dpop_loss,
}


def get_loss(name: str):
    try:
        return LOSS_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown loss '{name}'. Registered: {sorted(LOSS_REGISTRY)}"
        ) from None
