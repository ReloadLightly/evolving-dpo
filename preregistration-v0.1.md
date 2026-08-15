# Preregistration v0.1 — DRAFT FOR ATTACK

**Whose judgement does a preference-tuned model learn: the field's standards, or the reviewers' prose?**

Status: draft written by Claude for Roland to attack. Nothing here is settled.
Date: 2026-08-14. Target: arXiv preprint + workshop submission by 2026-08-31.

---

## 0. The one-sentence question

Claude's version, to be replaced by Roland's:

> When you preference-tune a language model on real peer-review decisions, does it
> learn what a research community counts as a good argument — or only how its
> reviewers write?

**Roland's version:** _______________________________________________

If the sentence in your own words is not better than mine, we are not ready to build.

---

## 1. Why this question, and why now

Three facts, each verified, that together make the question worth asking.

**(a) Communities really do apply different standards.** "Reviewer Scores Are Not
Comparable Across Research Areas" ([arXiv:2607.27209](https://arxiv.org/abs/2607.27209))
covers ICLR 2021–2026, 50,289 papers, 219 topics. At an *identical* reviewer score,
acceptance probability varies by up to **8×** depending on the paper's topic.
Acceptance thresholds span 0.812 rating points. Topic identity adds explanatory power
beyond the score itself (χ² = 341.2, p = 5.1e-3), and differences in scoring culture
account for only 21% of the gap. So the thing we want a model to learn demonstrably
exists in the data.

**(b) Preference pairs built from provenance teach style, not substance.** BridgeAlign
([arXiv:2607.27366](https://arxiv.org/abs/2607.27366)) tested the naive design — human
text is *chosen*, model text is *rejected* — and it **underperformed the untuned base
model** (60.40 vs 60.70). Their diagnosis: the model exploits superficial source cues.
Any design where the two sides of a pair differ in genre is suspect by default.

**(c) Preference optimization compresses exactly the distinctiveness we want to measure.**
PLURAL ([arXiv:2607.08034](https://arxiv.org/abs/2607.08034)) found DPO retains ~18% of
ground-truth cross-country value variation where SFT retains 30%. If the same holds for
scientific communities, DPO may be the wrong instrument — which is itself a finding.

**Metascience framing.** Alexander Krauss, *The Engine of Scientific Discovery* (OUP,
Feb 2026), argues from a survey of 750+ breakthroughs that discovery is driven by new
methods and instruments rather than by chance or paradigm shift. Peer review is the
institution that decides which of those claims counts. A model that could be shown to
encode a community's acceptance standard would be a measurement instrument for that
standard. That is the ambition; this preregistration tests only whether the first step
works at all.

## 2. What is already occupied — read this before you get attached

**RbtAct** ([arXiv:2603.09723](https://arxiv.org/abs/2603.09723), March 2026) already
built `REVIEWPREF-DPO-22K`: 21,822 chosen/rejected pairs of **real review text**, from
ICLR 2024, **constrained to the same paper and same perspective specifically to control
for confounds**, then SFT + DPO on Llama-3.1-8B.

Our style-matching argument is their published design constraint. We do not get to
claim it.

What RbtAct does *not* do, and what is therefore still ours:

| | RbtAct | This study |
|---|---|---|
| Who arbitrates the pair | the **author** (did the rebuttal address it) | the **institution** (did the decision follow it) |
| Target | actionability of feedback | epistemic standard of a community |
| Style-vs-substance probe | none | required, and pre-specified below |
| Cross-community design | none | the central hypothesis |

Also relevant and non-overlapping: Re² ([arXiv:2505.07920](https://arxiv.org/abs/2505.07920))
supplies the corpus landscape; CycleReviewer ([arXiv:2411.00816](https://arxiv.org/abs/2411.00816))
uses review scores as a reward for *paper* generation, not for adjudicating between
reviews; MReD, DISAPERE, ARIES, PeerRead, NLPeer are all non-preference tasks.

**No prior work was found that builds preference pairs from the institution's own
verdict, and none that runs a cross-community transfer experiment with review models.**

## 3. Hypotheses

Each states a prediction, and what result would falsify it. Written before any data is
touched.

**H1 — Learnability.** A model preference-tuned on institutionally-arbitrated review
pairs identifies, on held-out papers, which of two divergent reviews the institution
followed, better than (i) the untuned base model, (ii) an SFT-on-chosen-only control,
and (iii) a surface-feature classifier (§6).
*Falsified if* the tuned model does not beat all three, with non-overlapping 95% CIs
across 5 seeds.

**H2 — Substance over style.** The advantage in H1 survives style controls: it persists
on a length- and sentiment-matched subsample, and it collapses on shuffled pairs
(reviews drawn from different papers).
*Falsified if* the advantage disappears under matching, or persists under shuffling.
Persistence under shuffling would mean the model learned a global "good review" prior
rather than anything paper-specific.

**H3 — Community specificity (the crossover; this is the contribution).** Let A and B be
two research areas with documented different acceptance thresholds. A model tuned on A
scores higher on A's held-out pairs than on B's, and a model tuned on B shows the
reverse. Formally: a significant tuning-set × test-set interaction.
*Falsified if* the interaction term is not significant, i.e. both models perform equally
everywhere. That null is informative: it would mean DPO cannot capture a difference that
[2607.27209] shows is present in the decision statistics.

**H4 — Optimizer sensitivity (exploratory, run only if time allows).** The size of the
H3 interaction differs between DPO and a length-normalized reference-free objective
(SimPO), i.e. the choice of preference objective changes how much community-specific
signal survives.
*Exploratory. No confirmatory claim will be made from it.*

## 4. Data

**Source.** OpenReview, ICLR 2020–2025. Reviews are published for rejected submissions
as well as accepted ones, which is what makes ICLR usable and NeurIPS not (NeurIPS
publishes reviews only for accepted papers; ICML does not publish reviews; ARR reviews
are closed). Terms of Use, last updated 2024-09-24: comments are released **CC BY 4.0**,
metadata **CC0**.

**Pool.** ICLR 2020–2025 = 32,789 submissions (2,594 / 2,997 / 3,391 / 4,938 / 7,304 /
11,565), roughly four reviews each. ICLR 2026 is **excluded**: organizers reset all
scores to a pre-rebuttal state mid-discussion after a security incident, so its record
is not a clean institutional trace.

**Pair construction.** For each submission whose reviews are *split* — at least one
recommendation on each side of the venue's accept threshold — form the pair:

- **chosen** = the review whose recommendation matched the final decision,
- **rejected** = the review on the losing side,
- **prompt** = the paper's title + abstract (+ optionally the first N tokens of the
  introduction), so both sides answer the same question.

Both texts are individual reviews of one paper, written in the same genre, in the same
review cycle, under the same instructions. The pair is style-matched by construction and
the arbiter is the institution.

Estimated yield: **8,000–12,000 pairs** (estimate, not measured — the first task in §10
is to measure it).

**Why not "the review the meta-review sided with."** Meta-reviews almost never name
reviewers, so that label needs an LLM adjudication pass, which imports judge noise into
the ground truth. We use the decision-following label for v1 because it is deterministic
and reproducible, and treat meta-review agreement as a v2 refinement.

**Why not pre/post-rebuttal pairs.** OpenReview does not publicly expose review edit
history; the only public source of initial-vs-final scores is
`papercopilot/iclr-insights` (ICLR 2024–2025, CC BY 4.0), which carries scores but not
pre-rebuttal *text*, and traces reviewer identity heuristically. Ruled out.

**Splits.** Both are structural, not random:
1. **Temporal**: train on 2020–2023, test on 2024–2025. A random split would let the
   model memorize venue-specific fashions of a single year.
2. **Topic**: for H3, split by primary area, using the area taxonomy of [2607.27209].
   Areas chosen for maximum documented threshold separation, selected **before** any
   model is trained and recorded here once measured.

## 5. Arms

| Arm | What it is | Purpose |
|---|---|---|
| `base` | Qwen3-0.6B, untouched | floor |
| `sft` | SFT on chosen reviews only | isolates the contribution of the *pairing* |
| `dpo` | DPO on the constructed pairs | the treatment |
| `dpo-shuffled` | DPO on pairs whose two sides come from different papers | destroys the institutional signal, keeps the style signal |
| `surface` | logistic regression on surface features | the baseline that decides H2 |

Model: Qwen3-0.6B + LoRA (r=16), which the existing `evolving-dpo` trainer already
supports. β tuned on a validation split for the `dpo` arm — the baseline gets its best
shot, not its default.

## 6. Metrics and the baselines that matter

**Primary metric.** Pairwise accuracy: given a held-out paper and its two divergent
reviews, does the model's implicit reward rank the institution-followed review higher?
This is exactly `pref_accuracy` in `evaluate.py`, already implemented.

**The baseline that will kill a careless version of this study.** Reviews recommending
rejection read differently from reviews recommending acceptance, and ~70% of ICLR
submissions are rejected. A model that always prefers the more negative review would
therefore score ~70% and look impressive.
**Mitigation, fixed in advance: the evaluation set is balanced 50/50 between accepted
and rejected papers**, so that strategy scores 50%. We report the always-negative
baseline explicitly in every table.

**Surface-feature classifier** (the H2 arbiter): logistic regression on review length,
number of questions, hedge-word count, sentiment polarity, and count of numeric/citation
mentions. If the tuned language model does not beat this, it learned style. Reporting
this baseline is not optional.

**Secondary metrics**, all already implemented in the repo: mean implicit-reward margin;
per-token drift from the reference model; `chosen_logp_shift` (likelihood displacement),
so we can see whether the tuned model becomes *less* likely to produce the reviews it
supposedly prefers.

## 7. Analysis plan

- **5 seeds per arm.** Report mean and 95% CI, not a single run.
- **H1/H2**: non-overlapping CIs against each named baseline. No p-value fishing;
  the comparisons listed above are the complete set.
- **H3**: a logistic regression on pair-level correctness with terms
  `tuning_area`, `test_area`, and their **interaction**. The interaction coefficient is
  the result. A main effect without an interaction does not support H3.
- **Effect sizes reported in accuracy points**, not only as significance.
- No metric is added after data inspection. Anything discovered later is labelled
  exploratory in the write-up.

## 8. Kill criteria — when we abandon

Stated now so that we cannot rationalize later.

1. **Yield too low.** Fewer than 3,000 usable split-review pairs after construction →
   stop; the study is underpowered and no amount of modelling fixes it.
2. **Baseline dominance.** If the surface-feature classifier matches the tuned model
   within its CI, we do not have a values result. We report that, and stop.
3. **Reproduction failure.** If the harness cannot reproduce a published DPO
   preference-accuracy figure on a standard set (UltraFeedback), the instrument is not
   trustworthy and nothing downstream is either.
4. **Time.** If the dataset is not built and validated by 2026-08-21, we ship the
   dataset and the preregistration as the artifact, without model results.

## 9. Threats to validity, named

- **Outcome-agreement is not correctness.** A review that matched the decision may have
  matched it by predicting the majority, not by being right. Mitigation: report accuracy
  separately on papers where the decision went *against* the numerical majority of
  reviewers. That subset is small but it is the honest test.
- **Reviewer quality is confounded with reviewer identity.** Some reviewers are simply
  more experienced. We cannot control for this; we name it.
- **Topic leakage.** Held-out areas may share vocabulary with training areas. Partly
  mitigated by the temporal split, not eliminated.
- **LLM-written reviews contaminate later years.** One estimate puts ~21% of recent ICLR
  reviews as AI-generated. This is a genuine and unfixable contaminant for 2024–2025;
  we report it as a limitation and, if time allows, run the H1 test on 2020–2022 only
  as a sensitivity check.
- **Population validity.** ICLR is one venue in one field. Nothing here generalizes to
  "science."

## 10. Work plan to 2026-08-31

| Days | Task | Exit condition |
|---|---|---|
| 1–2 | Pull ICLR 2020–2025 via `openreview-py` 2.4.2; **measure** the real pair yield | a number replaces the 8–12k estimate |
| 2–3 | Build and release the pair dataset + construction script | dataset card, licence, counts by year and area |
| 3 | Reproduce a published DPO preference-accuracy figure on UltraFeedback | harness validated (kill criterion 3) |
| 4–7 | Train all five arms × 5 seeds | results table with CIs |
| 7–9 | H3 crossover: two areas × two tuned models | interaction coefficient |
| 9–12 | Write-up | 4–8 page workshop paper |
| 12–15 | Buffer, sensitivity checks, arXiv | preprint posted |

**Practical blocker, found today:** this cloud sandbox cannot reach `api.openreview.net`
(403 at the proxy). The data pull has to run on Roland's machine. Everything downstream
can run anywhere.

## 11. Ethics

Reviews on OpenReview are public and CC BY 4.0, and RbtAct and Re² establish precedent
for research use. But reviewers wrote them for authors and area chairs, not as training
data for preference optimization. We commit to:

- releasing pair *identifiers* and construction code rather than a bulk redistribution
  of review text, so the licence chain stays intact;
- no reviewer de-anonymization, and no per-reviewer analysis;
- an explicit statement that the artifact must not be used in a deployed reviewing
  system — the Review-5K licence takes the same position, and the 2026 literature on
  automating peer review ([arXiv:2605.03202](https://arxiv.org/abs/2605.03202)) argues
  the evaluation standards for that do not yet exist.

## 12. What we release

Code, the pair-construction script, dataset identifiers, all five arms' configs, all
seeds, all logs including divergent runs, and this preregistration with its git history
so the timestamps are checkable.

---

## Open decisions — Roland's, not Claude's

1. **The sentence in §0.** If it is not yours, the project is not yours.
2. **Which two areas for H3.** Maximum documented threshold separation is one criterion;
   your own judgement about which two communities genuinely differ epistemically is
   another, and probably better.
3. **Repo.** This is not "evolving dpo" any more. New repo with the trainer copied in,
   or a new branch of the old one?
4. **Whether H4 (the ES/optimizer angle) stays.** Dropping it makes the paper cleaner.
   Keeping it preserves the link to the neuroevolution thread you started from.
