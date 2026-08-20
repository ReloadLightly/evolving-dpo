"""The surface-feature classifier: the baseline that decides H2.

If a logistic regression on review length, question count, hedge words,
sentiment and citation counts can pick the decision-side review as well as the
tuned language model can, then the language model learned style, not judgement.
Reporting this baseline is not optional -- it is the arbiter, and a study that
omits it cannot claim to have measured substance.

Deliberately hermetic: numpy only, no sklearn, no downloads, fixed iteration
count and no randomness beyond an explicit seed. The lexicons are frozen in
this file so the features are auditable and identical on every machine.

The pairwise setup
------------------
A ranking task, not a classification of single reviews. For each pair the two
reviews are presented in a seeded random order as (first, second); the feature
vector is f(first) - f(second) and the label is 1 when `first` is the
decision-side review. Randomizing the order matters: without it every label
would be 1 and the model would learn the constant, scoring 100% while
measuring nothing.
"""

from __future__ import annotations

import math
import random
import re
from typing import Sequence

import numpy as np

HEDGES = (
    "may", "might", "could", "possibly", "perhaps", "seems", "appears",
    "suggests", "unclear", "somewhat", "arguably", "presumably", "likely",
    "unsure", "questionable", "potentially", "apparently",
)

POSITIVE = (
    "novel", "strong", "clear", "elegant", "thorough", "solid", "convincing",
    "interesting", "well-written", "impressive", "significant", "compelling",
    "rigorous", "useful", "excellent", "sound", "insightful",
)

NEGATIVE = (
    "unclear", "weak", "incremental", "limited", "flawed", "confusing",
    "insufficient", "lacking", "missing", "trivial", "unconvincing", "poor",
    "concern", "problem", "wrong", "misleading", "questionable", "fails",
)

FEATURE_NAMES = (
    "n_words", "n_sentences", "mean_sentence_len", "n_questions",
    "n_hedges", "hedge_rate", "n_positive", "n_negative", "sentiment",
    "n_numbers", "n_citations", "n_sections",
)

_CITATION = re.compile(r"\[\d+\]|\(\s*[A-Z][A-Za-z\-]+\s+(?:et al\.?,?\s*)?\d{4}\s*\)")
_NUMBER = re.compile(r"(?<![\w])\d+(?:\.\d+)?(?![\w])")
_WORD = re.compile(r"[A-Za-z']+")


def features(text: str) -> np.ndarray:
    """Twelve surface features. No semantics, by design."""
    text = text or ""
    low = text.lower()
    words = _WORD.findall(low)
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = len(sentences)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    n_hedges = sum(counts.get(h, 0) for h in HEDGES)
    n_pos = sum(counts.get(p, 0) for p in POSITIVE)
    n_neg = sum(counts.get(n, 0) for n in NEGATIVE)
    denom = max(1, n_pos + n_neg)

    return np.array([
        n_words,
        n_sentences,
        n_words / max(1, n_sentences),
        text.count("?"),
        n_hedges,
        n_hedges / max(1, n_words),
        n_pos,
        n_neg,
        (n_pos - n_neg) / denom,          # polarity in [-1, 1]
        len(_NUMBER.findall(text)),
        len(_CITATION.findall(text)),
        text.count("##"),                  # section headings, a format cue
    ], dtype=np.float64)


def build_design(pairs: Sequence[dict], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Feature differences and labels, with the presentation order randomized."""
    rng = random.Random(seed)
    rows, labels = [], []
    for pair in pairs:
        chosen, rejected = pair.get("chosen"), pair.get("rejected")
        if not chosen or not rejected:
            continue
        f_chosen, f_rejected = features(chosen), features(rejected)
        if rng.random() < 0.5:
            rows.append(f_chosen - f_rejected)
            labels.append(1.0)
        else:
            rows.append(f_rejected - f_chosen)
            labels.append(0.0)
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    return np.vstack(rows), np.array(labels)


class SurfaceClassifier:
    """L2-regularized logistic regression, full-batch gradient descent.

    Small and deterministic on purpose: this baseline has to be reproducible
    byte-for-byte, and at a dozen features and a few thousand rows there is no
    reason to reach for an optimizer anyone has to trust.
    """

    def __init__(self, l2: float = 1.0, lr: float = 0.5, n_iter: int = 2000):
        self.l2, self.lr, self.n_iter = l2, lr, n_iter
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mean_ = X.mean(axis=0)
            scale = X.std(axis=0)
            scale[scale < 1e-8] = 1.0        # a constant feature carries nothing
            self.scale_ = scale
        return (X - self.mean_) / self.scale_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurfaceClassifier":
        if len(X) == 0:
            raise ValueError("no training rows")
        Xs = self._standardize(X, fit=True)
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w / n
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.scale_
        return 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return float("nan")
        return float(((self.predict_proba(X) >= 0.5).astype(float) == y).mean())

    def weights(self) -> dict[str, float]:
        """Standardized coefficients, largest magnitude first.

        Worth reading rather than just the accuracy: if `n_words` dominates,
        the baseline is measuring length and so, probably, is the tuned model.
        """
        if self.w is None:
            return {}
        pairs = zip(FEATURE_NAMES, self.w.tolist())
        return dict(sorted(pairs, key=lambda kv: -abs(kv[1])))


def evaluate_surface(train_pairs: Sequence[dict], test_pairs: Sequence[dict],
                     seed: int = 0, l2: float = 1.0) -> dict:
    """Fit on train, score on test. The H2 arbiter, end to end."""
    X_tr, y_tr = build_design(train_pairs, seed=seed)
    X_te, y_te = build_design(test_pairs, seed=seed + 1)
    if len(X_tr) == 0 or len(X_te) == 0:
        return {"accuracy": float("nan"), "n_train": len(X_tr), "n_test": len(X_te)}
    clf = SurfaceClassifier(l2=l2).fit(X_tr, y_tr)
    return {
        "accuracy": clf.accuracy(X_te, y_te),
        "train_accuracy": clf.accuracy(X_tr, y_tr),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "weights": clf.weights(),
        "seed": seed,
    }
