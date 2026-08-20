"""Splits, arms and trivial baselines for the peer-review preference study.

`scripts/build_pairs.py` produces `data/pairs.jsonl`; the trainer in this
package consumes `{"prompt", "chosen", "rejected"}`. This module is the bridge,
and it also implements the three things the study needs that are not training:
structural splits, the 50/50 balancing that makes the numbers mean anything,
and the two trivial baselines every results table has to report.

Nothing here imports torch, so all of it is testable on a laptop.

The 70% problem, concretely
---------------------------
About 71% of pairs come from rejected papers, and for a rejected paper the
`chosen` review is the negative one. So "always prefer the more negative
review" is right on every rejected paper: on the raw corpus it scores ~71% and
looks like a result. `balance_accept_reject` makes it score 50% instead, and
`baseline_always_negative` reports what it actually scored so the number is
visible rather than assumed. Both are required, not optional.
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
from typing import Any, Iterable, Sequence

# Broad areas, defined by keyword. Frozen here deliberately: H3's area split
# must be fixed before training, and SNOR ships no `primary_area` at all
# (OpenReview only has it from 2024), so for earlier years an area has to be
# induced. Keeping the lexicon in version control makes the induction auditable
# and reproducible, which clustering on embeddings would not be without also
# committing the embeddings and the seed.
AREA_LEXICON: dict[str, tuple[str, ...]] = {
    "theory": (
        "generalization bound", "convergence", "optimization theory", "pac",
        "sample complexity", "regret", "convex", "statistical learning",
        "information theory", "provable", "theoretical analysis",
    ),
    "reinforcement_learning": (
        "reinforcement learning", "policy gradient", "q-learning", "bandit",
        "markov decision", "actor-critic", "exploration", "offline rl", "agent",
        "imitation learning", "reward",
    ),
    "vision": (
        "computer vision", "image", "segmentation", "object detection", "video",
        "convolutional", "visual", "3d", "point cloud", "diffusion model",
        "image generation",
    ),
    "language": (
        "language model", "nlp", "natural language", "text", "translation",
        "transformer", "question answering", "summarization", "tokenizer",
        "instruction tuning", "llm",
    ),
    "graph": (
        "graph neural", "gnn", "graph representation", "node classification",
        "link prediction", "message passing", "knowledge graph",
    ),
    "generative": (
        "generative model", "gan", "variational autoencoder", "vae",
        "normalizing flow", "score-based", "diffusion", "sampling",
    ),
    "trustworthy": (
        "fairness", "privacy", "differential privacy", "robustness",
        "adversarial", "interpretability", "explainability", "calibration",
        "uncertainty", "safety", "alignment",
    ),
    "efficiency": (
        "quantization", "pruning", "distillation", "sparsity", "efficient",
        "compression", "federated", "hardware", "inference speed",
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_pairs(path: str | pathlib.Path,
               years: Iterable[int] | None = None) -> list[dict]:
    """Read pairs.jsonl, optionally restricted to certain years."""
    path = pathlib.Path(path)
    wanted = set(years) if years is not None else None
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if wanted is not None and rec.get("year") not in wanted:
                continue
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------

def assign_area(pair: dict, lexicon: dict[str, tuple[str, ...]] | None = None) -> str | None:
    """Best-matching area for one pair, or None if nothing matches.

    Prefers the venue's own `primary_area` when present (2024+), falling back
    to keyword induction. Ties go to the alphabetically first area so the
    assignment is deterministic across runs and machines.
    """
    lexicon = lexicon or AREA_LEXICON

    primary = (pair.get("primary_area") or "").strip().lower()
    haystacks = []
    if primary:
        haystacks.append(primary)
    keywords = pair.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    haystacks.extend(str(k).lower() for k in keywords)
    # Title and abstract are a last resort; the prompt holds both.
    haystacks.append(str(pair.get("prompt") or "").lower()[:1000])
    blob = " | ".join(haystacks)

    scores = {
        area: sum(1 for term in terms if term in blob)
        for area, terms in lexicon.items()
    }
    best = max(sorted(scores), key=lambda a: scores[a])
    return best if scores[best] > 0 else None


def assign_areas(pairs: Sequence[dict],
                 lexicon: dict[str, tuple[str, ...]] | None = None) -> tuple[list[dict], dict]:
    """Attach `area` to every pair; return the pairs and a coverage report.

    The report exists so an area split is never chosen without seeing how many
    pairs it actually covers -- H3's power is set by the smallest cell.
    """
    out, counts, unassigned = [], collections.Counter(), 0
    for pair in pairs:
        area = assign_area(pair, lexicon)
        enriched = dict(pair)
        enriched["area"] = area
        out.append(enriched)
        if area is None:
            unassigned += 1
        else:
            counts[area] += 1
    report = {
        "n_pairs": len(pairs),
        "n_assigned": len(pairs) - unassigned,
        "n_unassigned": unassigned,
        "coverage": (len(pairs) - unassigned) / len(pairs) if pairs else 0.0,
        "by_area": dict(counts.most_common()),
    }
    return out, report


# --------------------------------------------------------------------------
# Balancing and splits
# --------------------------------------------------------------------------

def balance_accept_reject(pairs: Sequence[dict], seed: int = 0) -> list[dict]:
    """Subsample to exactly 50/50 accepted vs rejected papers.

    Without this, "always prefer the more negative review" scores ~71%. The
    majority class is subsampled with a seeded RNG, so the same corpus and seed
    always give the same evaluation set.
    """
    accepted = [p for p in pairs if p.get("accepted")]
    rejected = [p for p in pairs if not p.get("accepted")]
    n = min(len(accepted), len(rejected))
    rng = random.Random(seed)
    accepted = sorted(accepted, key=lambda p: str(p.get("paper_id")))
    rejected = sorted(rejected, key=lambda p: str(p.get("paper_id")))
    out = rng.sample(accepted, n) + rng.sample(rejected, n)
    out.sort(key=lambda p: (p.get("year", 0), str(p.get("paper_id"))))
    return out


def temporal_split(pairs: Sequence[dict],
                   train_years: Iterable[int] = (2020, 2021, 2022, 2023),
                   test_years: Iterable[int] = (2024, 2025)) -> tuple[list[dict], list[dict]]:
    """Split by year. Structural, never random: a random split would let the
    model memorize a single year's venue fashions."""
    train_years, test_years = set(train_years), set(test_years)
    train = [p for p in pairs if p.get("year") in train_years]
    test = [p for p in pairs if p.get("year") in test_years]
    return train, test


def topic_split(pairs: Sequence[dict], area_a: str, area_b: str) -> dict[str, list[dict]]:
    """Partition into the two areas of the H3 crossover.

    Returns {"a": ..., "b": ...}. Both models are later scored on BOTH sets,
    which is why per-cell N is the size of one area rather than a quarter of
    the corpus.
    """
    if area_a == area_b:
        raise ValueError("H3 needs two distinct areas")
    return {
        "a": [p for p in pairs if p.get("area") == area_a],
        "b": [p for p in pairs if p.get("area") == area_b],
    }


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def to_preference_examples(pairs: Sequence[dict]) -> list[dict]:
    """The `dpo` arm: {"prompt", "chosen", "rejected"} for encode_pair."""
    return [
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
        if p.get("prompt") and p.get("chosen") and p.get("rejected")
    ]


def to_sft_examples(pairs: Sequence[dict]) -> list[dict]:
    """The `sft` arm: chosen reviews only.

    Isolates what the *pairing* contributes: same prompts, same preferred
    text, no contrast to learn from.
    """
    return [
        {"prompt": p["prompt"], "completion": p["chosen"]}
        for p in pairs
        if p.get("prompt") and p.get("chosen")
    ]


def shuffled_pairs(pairs: Sequence[dict], seed: int = 0) -> list[dict]:
    """The `dpo-shuffled` arm: `rejected` replaced from a DIFFERENT paper.

    Destroys the paper-specific institutional signal while leaving the generic
    style signal intact. If the advantage survives this, the model learned a
    global "good review" prior rather than anything about the paper -- which is
    H2's falsification condition.

    Derangement by rotation, so no pair can keep its own rejected review even
    when the corpus is tiny.
    """
    usable = [p for p in pairs if p.get("prompt") and p.get("chosen") and p.get("rejected")]
    if len(usable) < 2:
        return []
    rng = random.Random(seed)
    order = list(range(len(usable)))
    rng.shuffle(order)
    out = []
    for i, idx in enumerate(order):
        donor = usable[order[(i + 1) % len(order)]]
        host = usable[idx]
        if donor["paper_id"] == host["paper_id"]:
            continue  # cannot borrow from itself
        out.append({
            "prompt": host["prompt"],
            "chosen": host["chosen"],
            "rejected": donor["rejected"],
            "paper_id": host["paper_id"],
            "donor_paper_id": donor["paper_id"],
            "year": host.get("year"),
            "accepted": host.get("accepted"),
        })
    return out


# --------------------------------------------------------------------------
# Trivial baselines -- report these in every table
# --------------------------------------------------------------------------

def baseline_always_negative(pairs: Sequence[dict]) -> float:
    """Accuracy of "always prefer the lower-rated review".

    By construction `chosen` is the lower-rated review exactly when the paper
    was rejected, so this equals the rejected fraction: ~71% on the raw corpus
    and 50% on a balanced one. That is the whole reason balancing is mandatory.
    """
    scored = [p for p in pairs
              if p.get("chosen_rating") is not None and p.get("rejected_rating") is not None]
    if not scored:
        return float("nan")
    return sum(1 for p in scored if p["chosen_rating"] < p["rejected_rating"]) / len(scored)


def baseline_always_longer(pairs: Sequence[dict]) -> float:
    """Accuracy of "always prefer the longer review" -- projected at 56.2%."""
    scored = [p for p in pairs
              if p.get("chosen_len_tokens") is not None
              and p.get("rejected_len_tokens") is not None]
    if not scored:
        return float("nan")
    return sum(1 for p in scored
               if p["chosen_len_tokens"] > p["rejected_len_tokens"]) / len(scored)


def against_majority_subset(pairs: Sequence[dict]) -> list[dict]:
    """Pairs where the decision went against the numerical reviewer majority.

    The honest test: here the label cannot have been guessed by predicting the
    majority. Excludes even splits, which have no majority to disagree with.
    """
    return [p for p in pairs if p.get("majority_agreed_with_decision") is False]


def summarize(pairs: Sequence[dict]) -> dict[str, Any]:
    """Everything a results table needs to state about a split."""
    n = len(pairs)
    accepted = sum(1 for p in pairs if p.get("accepted"))
    return {
        "n_pairs": n,
        "n_accepted": accepted,
        "n_rejected": n - accepted,
        "accept_rate": accepted / n if n else float("nan"),
        "years": sorted({p.get("year") for p in pairs if p.get("year") is not None}),
        "areas": dict(collections.Counter(
            p.get("area") for p in pairs if p.get("area")).most_common()),
        "n_against_majority": len(against_majority_subset(pairs)),
        "baseline_always_negative": baseline_always_negative(pairs),
        "baseline_always_longer": baseline_always_longer(pairs),
        "baseline_random": 0.5,
    }
