# Institutional preference pairs from peer review

**When two reviewers of the same paper disagree, the venue's decision falls on one
side. Can a preference-tuned model predict which — and is that signal specific to a
research community, or just the surface style of reviews?**

Peer review is an institution that records decisions about competing judgements. For a
split ICLR submission, one reviewer argues for acceptance and another for rejection,
both writing about the same paper, in the same genre, in the same cycle, under the same
instructions. The programme committee then rules. Pairing the two reviews and labelling
them by which side the ruling fell on gives a preference dataset whose label comes from
the venue's own record rather than from an author or an LLM judge. Across ICLR
2020–2024 that yields a projected **~10,000 pairs** — about 6,500 for training and
2,202 for testing after the mandatory 50/50 balancing — studied at 0.6B scale.

**Status, 2026-08-20. No model has been trained. No dataset has been built. No review
text has ever been fetched.** This repository contains the construction pipeline, the
data layer and the surface baseline with 135 offline tests, a timestamped
preregistration, a power analysis, and a yield projection computed from real
ICLR ratings with placeholder text. Every number below
carries a label saying which of those it is; none is an experimental result.

**Clone note.** This work lives on branch `claude/openreview-fetch-reviews-j3jee5`.
`main` still ends at the pre-peer-review mini-lab and contains none of it. The repo also
hosts that earlier project — a small evolutionary-DPO lab whose trainer this study
reuses; see [Relationship to the evolving-DPO work](#relationship-to-the-evolving-dpo-work),
which answers plainly whether evolution gets used here.

---

## The construction rule

For each ICLR submission whose reviewer ratings **straddle that year's accept
threshold**:

- `chosen` — the most extreme review on the side the venue's decision took
- `rejected` — the most extreme review on the losing side
- `prompt` — the paper's title and abstract, so both sides answer the same question

Three details carry weight:

**The threshold is estimated per year, never hardcoded.** It is the rating cut
maximizing Youden's J against that year's real accept/reject outcomes (Youden rather
than accuracy, because with ~70% rejections accuracy is maximized by a degenerate cut).
This is not fussiness: ICLR 2020 rates on {1,3,6,8}, 2021 on 1–10, and 2022 onwards on
{1,3,5,6,8,10}. One fixed cut of 5 would mean a different thing every year.

**Ties break on review id, never on length.** Preferring the longer review would build
the length confound directly into the labels that H2 exists to rule out.

**A minimum rating spread** (`--min-spread`, default 1.0) is applied on top of the
straddle requirement. Note this is looser than the ≥3 spread used in the Task 1 yield
measurement, so those two counts are not directly comparable.

### What the label is, and what it is not

The label is **a proxy, not an adjudication.** The programme committee rules on the
paper, not on the reviewers, and it may rule for reasons neither review contains. What
the data records is which review fell on the side the decision took. We adopt it for v1
because it is deterministic, automatic and checkable. The label that would directly
encode "which reviewer the area chair sided with" is ruled out in HANDOFF §7: meta-reviews
almost never name reviewers, so recovering it needs an LLM adjudication pass, which
imports judge noise into the ground truth.

Likewise, the pairs are matched on **paper, genre, review cycle and reviewer
instructions — but not on stance.** By construction the two sides sit on opposite sides
of the accept threshold and are the most extreme review on each, so they differ
systematically in valence, hedging and severity. That residual is the study's principal
confound, not a controlled variable, and it is why the always-negative baseline, the
always-longer baseline and the surface-feature classifier all have to be reported.

### What is new here

Same-paper style matching is not new — it is
[RbtAct](https://arxiv.org/abs/2603.09723)'s published design constraint, and they have
21,822 real-text pairs and a trained 8B model. The differences are:

1. **The arbiter.** RbtAct's is the author (did the rebuttal address the point); here it
   is the venue's recorded decision. This is the line for the abstract.
2. **The experiment that arbiter makes possible** — a cross-community crossover (H3)
   testing whether the preference signal is area-specific.
3. **Likelihood-displacement diagnostics** in this setting, which the inherited trainer
   already computes.

### Hypotheses

Full statements with falsification conditions are in
[preregistration-v0.1.md](preregistration-v0.1.md).

| | Claim | Falsified if |
|---|---|---|
| **H1** | A DPO-tuned model picks the decision-side review on held-out papers better than the untuned base, an SFT-on-chosen-only control, and a surface-feature classifier | it fails to beat all three with non-overlapping 95% CIs over 5 seeds |
| **H2** | The H1 advantage survives length/sentiment matching and collapses on shuffled cross-paper pairs | the advantage vanishes under matching, or persists under shuffling |
| **H3** | *(the contribution)* A model tuned on area A scores higher on A's held-out pairs than on B's, and vice versa — a significant `tuning_area × test_area` interaction | the interaction is not significant |
| **H4** | *(exploratory)* The H3 interaction differs in size between DPO and a reference-free length-normalized objective | n/a — no confirmatory claim |

H3's **premise** is not a guess. "Reviewer Scores Are Not Comparable Across Research
Areas" ([arXiv:2607.27209](https://arxiv.org/abs/2607.27209)) covers ICLR 2021–2026,
50,289 papers, 219 topics: at an identical reviewer score, acceptance probability varies
up to 8× by topic, thresholds span 0.812 rating points, and scoring culture explains
only 21% of the gap. H3's **magnitude in pairwise-accuracy points is entirely unknown** —
that paper reports acceptance probabilities and thresholds, not accuracy points, and
nothing here predicts an effect size. The power analysis is therefore a floor on what is
detectable, not a test of an expected value.

A null interaction would be reported with its CI and read narrowly. It could mean DPO
cannot capture the difference; it could equally mean the study is underpowered, the area
basis is wrong, or 0.6B is too small.

---

## Measured, projected, and not run

Nothing below should be quoted without its label.

| | Status |
|---|---|
| Yield table (`data/iclr_split_stats.json`) | **Measured** 2026-08-14 from `papercopilot/paperlists`: 21,392 submissions, of which 21,095 carry ≥2 ratings; 9,586 of those have spread ≥ 3 (45.4% of rated papers, 44.8% of all submissions) |
| Pair counts, thresholds, balance, length (`data/YIELD_PROJECTION.md`) | **Projected.** Real per-reviewer ratings and real decisions run through the real construction code — with *placeholder review text* padded to the real word count |
| Detectable effects (`results/power.md`) | **Computed** and reproducible — but from the projected counts above |
| Any model result | **Does not exist.** No arm trained, no `results/main.csv`, no baseline number |

### Projected yield, ICLR 2020–2024

From `scripts/validate_on_paperlists.py`, which runs `estimate_threshold` and
`make_pair` unchanged over real `papercopilot/paperlists` records:

```bash
git clone https://github.com/papercopilot/paperlists ~/paperlists
python scripts/validate_on_paperlists.py --paperlists ~/paperlists/iclr
```

| Year | Papers | Pairs | Yield | Accept threshold | Youden J |
|---|---|---|---|---|---|
| 2020 | 2,594 | 1,125 | 43.4% | 5.22 | 0.782 |
| 2021 | 3,014 | 1,613 | 53.5% | 5.90 | 0.765 |
| 2022 | 3,422 | 1,553 | 45.4% | 5.71 | 0.771 |
| 2023 | 4,955 | 2,220 | 44.8% | 5.69 | 0.798 |
| 2024 | 7,407 | 3,488 | 47.1% | 5.69 | 0.739 |
| **Total** | | **9,999** | | | |

**9,999 is a ceiling, not an estimate.** Three exclusion classes — `missing_text`,
`identical_text`, `no_prompt` — are zero for every year only because the run used
generated placeholder text. Real prose will trigger all three.

On those projected pairs: accepted **28.9%** vs rejected **71.1%**; the decision went
**against the reviewer majority** in **18.1%** (1,809); mean chosen review 486 words vs
rejected 434, with **chosen the longer review in 56.2%** of pairs.

ICLR 2025 is absent from every count here — the git-LFS fetch was blocked when Task 1
ran. ICLR 2026 is excluded by design: the organizers reset all scores to a pre-rebuttal
state mid-discussion after a security incident, so its record is not a clean
institutional trace.

### Power analysis (`results/power.md`)

Run before training, deliberately, so the design could still change. α = 0.05, 80%
power, normal approximations, on the projected counts.

Balancing 50/50 on accept/reject is non-negotiable — otherwise "always prefer the
negative review" scores ~70% — and the minority class binds:

| | Pairs |
|---|---|
| Training (2020–2023) | 6,511 |
| Test (2024), raw | 3,488 |
| Test after 50/50 balancing | **2,202** (−37%) |
| Of those, against the reviewer majority | ~362 |

*This is 2024 only. The preregistered temporal split tests on 2024–2025; ICLR 2025 was
never retrieved, so every figure below runs on roughly half the intended test set.*

**H1, pair-level.** Every arm scores the same pairs, so the paired row governs: **2.98 pp**
detectable at 25% discordance (4.10 pp if treated as independent samples). On the
against-majority subset, 7.34 pp paired — small but usable.

**H1, seed-level.** The preregistration's actual criterion — non-overlapping 95% CIs
over 5 seeds — is a test on five numbers, so between-seed SD governs and dataset size
does not enter. It is 1.23× stricter than a two-sample t-test, and corresponds to about
p = 0.006 rather than p = 0.05. At 2 pp seed SD, no amount of test data resolves a 3 pp
effect. Measure seed SD on the first arm trained.

**H3, the binding constraint.** On 2024 alone, ~1,101 pairs per test area detects an
interaction of **8.25 pp**. Using all years instead gives **4.97 pp**, because a topic
split holds out areas rather than years and may draw on the whole corpus. The recommended
design is therefore the all-years topic split, reported separately from the temporal
result.

**That recommendation is contingent.** `primary_area` exists in OpenReview only from
2024 and not at all in SNOR, so areas for 2020–2023 do not currently exist and must be
induced from `keywords` or by clustering SNOR's Specter embeddings. Until that basis is
built and frozen in git, 4.97 pp is the power of a design that cannot yet be run.

---

## Five ways to get a number that looks like a result and is not

1. **The 70% trap.** ~70% of submissions are rejected and rejection-recommending reviews
   read differently, so "always prefer the negative review" scores ~70%. Mitigation:
   balance the evaluation set 50/50 and report the always-negative baseline in every table.
2. **Length.** The always-longer-review baseline projects to **56.2%**, not 50%. Report
   it beside the always-negative one.
3. **Agreement is not correctness.** A review matching the decision may have matched it
   by predicting the majority. The against-the-majority subset is the honest test, which
   is why every row carries `majority_agreed_with_decision`.
4. **Style is the default thing to learn.** BridgeAlign
   ([arXiv:2607.27366](https://arxiv.org/abs/2607.27366)) built provenance-based pairs
   and *underperformed the untuned base model*, 60.40 vs 60.70, by exploiting source
   cues. Same-paper pairing is the partial defence — partial because the two sides still
   differ in stance — and the surface-feature classifier is what tests it.
5. **DPO may compress the very thing H3 measures.** PLURAL
   ([arXiv:2607.08034](https://arxiv.org/abs/2607.08034)) found DPO retains ~18% of
   cross-country value variation where SFT retains 30%. If the H3 interaction is weak,
   this is a candidate explanation and the SFT arm is the comparison that tests it.

---

## Running the pipeline

Prerequisites: Python 3.11. The data pipeline, the splits and the surface baseline
need only `pytest` and `numpy` — no torch, no network. `requirements.txt` additionally
pins torch and friends, which are needed for training only.

```bash
pip install pytest numpy
pytest tests/test_fetch_openreview.py tests/test_build_pairs.py \
       tests/test_load_snor.py tests/test_power_analysis.py \
       tests/test_review_pairs.py tests/test_surface_baseline.py -q   # 135 tests
```

**Route A — from SNOR (recommended).** SNOR v1 is a normalized OpenReview dump
(DOI [10.5281/zenodo.15866613](https://zenodo.org/records/15866613), CC BY 4.0; ICLR
2017–2025 and NeurIPS 2021–2025). Download and unpack it to `~/snor`, giving
`normalized_papers.jsonl`, `normalized_comments.jsonl` and `failed_matches.jsonl`
(~2.2 GB).

```bash
python scripts/inspect_snor.py ~/snor          # schema report, a few KB
python scripts/load_snor.py    ~/snor --include-failed
python scripts/build_pairs.py  --years 2020-2025
python scripts/power_analysis.py --index data/pairs_index.csv   # redo on real counts
```

`--include-failed` is not optional in spirit. SNOR links submissions to Semantic
Scholar, and the 8,541 that failed are overwhelmingly rejected and withdrawn papers —
rejected work often never gets published, so it has no Semantic Scholar record. The
unmatched set is led by "ICLR 2025 Rejected" (2,212) and "ICLR 2025 Withdrawn" (1,545)
against "ICLR 2025 Poster" (134). Since a projected ~71% of pairs come from rejected
papers, dropping them would skew the corpus toward rejects that were published elsewhere
later. `load_snor.py` prints a matched-vs-unmatched accept-rate table so the bias is
measured rather than assumed.

**Route B — from OpenReview directly.** Slower, but it is the source of record and the
way to spot-check SNOR's normalization.

```bash
pip install openreview-py
python scripts/probe_openreview.py                            # ~2 min sanity check
python scripts/fetch_openreview.py --check
python scripts/fetch_openreview.py --years 2023 --limit 25    # smoke test
python scripts/fetch_openreview.py --years 2020-2025
python scripts/build_pairs.py --years 2020-2025
```

The machine that wrote this code could reach neither `api.openreview.net` nor
`zenodo.org` (both 403 at an egress proxy). **That is specific to the authoring sandbox
— on an ordinary machine nothing here is blocked.** Whoever runs Route A first replaces
every projection in this repository with a measurement.

---

## What is in the repository

| Path | What it does |
|---|---|
| `scripts/load_snor.py` | SNOR dump → `data/raw/iclr{year}.jsonl`, with the selection-effect report |
| `scripts/fetch_openreview.py` | OpenReview API v1+v2 → the same records; caches raw JSON per year |
| `scripts/build_pairs.py` | raw records → `data/pairs.jsonl`, `pairs_index.csv`, `DATASET_CARD.md` |
| `scripts/validate_on_paperlists.py` | validates the construction logic on real ratings without review text |
| `scripts/power_analysis.py` | minimum detectable effects → `results/power.md` |
| `scripts/inspect_snor.py`, `probe_openreview.py` | schema reconnaissance before a long download |
| `src/evolving_dpo/review_pairs.py` | splits (temporal, topic), 50/50 balancing, the three arms' data, area induction, trivial baselines |
| `src/evolving_dpo/surface_baseline.py` | the surface-feature classifier that arbitrates H2, numpy only |
| `src/evolving_dpo/` (rest) | inherited LoRA DPO trainer, five losses, evaluation with drift and displacement diagnostics |
| `HANDOFF.md`, `preregistration-v0.1.md` | the plan and the timestamped hypotheses |

## What is missing, and what is broken

Stated plainly, because HANDOFF §12.1 makes the code — not the results — the deliverable.

**Still missing.** `scripts/run_all.py` (one command, all arms, 5 seeds),
`scripts/analyze.py` (tables and the three figures), and `paper/`. The H3 interaction
model is sized by `power_analysis.py` but not fitted. The `dpo` and `sft` arms need a
training entry point that consumes `review_pairs.to_preference_examples` /
`to_sft_examples` — the data side of both exists and is tested; the training side does
not.

**Broken in the inherited trainer**, verified by reading the code:

- `evaluate.py:144` computes the margin as `β·((pol_c − pol_r) − (ref_c − ref_r))`, so
  `pref_accuracy` is **reference-relative**. At LoRA initialization policy equals
  reference exactly, every margin is 0, and the `base` arm scores **0.0, not ~50%** —
  meaning H1's primary comparison is not valid as implemented. An absolute variant
  (`pol_c > pol_r`) must be added. Note preregistration §6 asserts this metric is
  "already implemented"; that assertion is wrong for the base arm.
- `trainer.py::sequence_logps` materializes a full-vocab log-softmax; at Qwen3's 151,936
  vocab and review-length sequences this will OOM a 16 GB GPU, and `run_candidate`
  records that OOM as candidate *divergence*.
- `load_policy` forces `.float()` for version-safety, costing roughly 5–7× on a T4
  versus fp16.

**Unvalidated.** Task 4 / kill criterion 3 — reproducing a published DPO
preference-accuracy figure on UltraFeedback — has never run. Everything downstream sits
behind an unvalidated instrument. The text-parsing half of the pipeline is exercised only
against hand-written fixtures.

**Open decisions** that belong to the project owner: the one-sentence research question
in his own words (still blank in HANDOFF §2.1), which two areas H3 uses, and whether H4
stays.

**Deadlines.** Preregistration kill criterion 4 is the live one: if the dataset is not
built and validated by **2026-08-21**, the commitment is to ship the dataset and the
preregistration alone, without model results. ARR closes 12 October 2026.

---

## Relationship to the evolving-DPO work

This repository began as a mini-lab for rediscovering preference-optimization objectives
by evolution, in the style of Sakana AI's [DiscoPOP](https://arxiv.org/abs/2406.08414),
driven by [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve). That lineage supplies
the trainer this study uses: LoRA policy with the reference obtained by disabling
adapters (so one model sits in memory), five interchangeable losses, and an evaluation
that reports drift and likelihood displacement alongside accuracy.

**Will evolution be used here? Not for this paper.** `shinka_task/` is still unarmed, and
a DiscoPOP-scale search is 30–60 candidates, each a training run. With no GPU, free-tier
Kaggle at 30 h/week, three of five arms unimplemented and an unvalidated harness,
spending that budget on search instead of the preregistered arms would trade a finishable
paper for an unfinishable one.

The connection worth keeping is sharper than the original one. `evaluate.py` already
computes a fitness — preference accuracy minus a drift penalty minus a displacement
penalty. But the quantity this project cares about is not preference accuracy; it is
**how much community-specific signal an objective preserves**, which is exactly what
PLURAL measured and left open. "Evolve a preference objective that maximally preserves
institutional distinctiveness" is a well-posed search and a different question from
DiscoPOP's "evolve an objective that maximizes preference accuracy". H4 is its
two-point, hand-run special case.

It is also not yet runnable as stated. The H3 interaction's standard error under the
recommended all-years design is roughly 1.8 pp before between-seed variance, so a 40–60
candidate search on that fitness would mostly select noise excursions. It needs either a
much larger corpus or a variance-reduced estimator first. Written down, not run.

## The dataset, and why there is not one here

Two separate reasons, and only one of them is a gap.

**Deliberate.** `data/raw/` and `data/pairs.jsonl` are gitignored. Reviews are CC BY 4.0
and were written for authors and area chairs, not as training data, so the release model
is **pair identifiers plus construction code** — `data/pairs_index.csv`, which carries
ids, ratings and labels but no prose — rather than a bulk redistribution. Anyone with
the same source snapshot can rebuild the text by running the committed code. Byte-exact
reproduction is not guaranteed: OpenReview reviews can be edited or withdrawn, and SNOR
v1 is a single dated dump.

**Not yet run.** `pairs_index.csv` and `DATASET_CARD.md` do not exist because nobody has
executed Route A. That is a few minutes of compute on a machine with the download, and
it is the single highest-value action available.

Known risks to the dataset, all documented rather than discovered later: the SNOR
Semantic-Scholar selection effect described above; the absence of `primary_area`, which
H3 needs; ICLR 2025 missing; ICLR 2026 excluded by design; and AI-generated review
contamination in recent years.

## Limitations

1. **One venue, one field, English only.** ICLR is not "science".
2. **Scale 0.6B**, with an 8B check planned rather than a scaling study.
3. **Agreement with the outcome is not correctness.** The against-the-majority subset is
   the response, and it is small (~362 balanced test pairs).
4. **The label is a proxy for institutional judgement**, not a record of one.
5. **The two sides differ in stance by construction**, which no amount of same-paper
   matching removes.
6. **Contamination.** One estimate puts ~21% of recent ICLR reviews as AI-generated
   *(source to be supplied)*. It lands in the test split; the 2020–2022 sensitivity
   analysis is a response, not a fix.
7. **Reviewer quality is confounded with reviewer identity.** We cannot control for it.
8. **The measuring instrument is unvalidated** — see kill criterion 3 above.
9. **Apart from the Task 1 yield measurement**, every number here is projected or
   computed from projections.

**We claim** — once the experiments run — that institutionally-labelled review pairs are
or are not learnable above named baselines; that the signal is or is not reducible to
surface style; and that it is or is not community-specific. **We do not claim** that the
model judges review quality, that it should be used in any reviewing system, that the
institution's decision was correct, or that anything generalizes beyond ICLR.

## Ethics

Reviews on OpenReview are public and CC BY 4.0; metadata is CC0.

**This artifact must not be used in a deployed reviewing system, in whole or in part.**
That includes the pair index, the construction code, and any model trained on pairs built
with it. The evaluation standards that would make automated reviewing defensible do not
yet exist — see "Stop Automating Peer Review Without Rigorous Evaluation"
([arXiv:2605.03202](https://arxiv.org/abs/2605.03202)).

We further commit to no reviewer de-anonymization, no per-reviewer analysis, and release
of identifiers plus code rather than bulk review text.

## Prior work and credit

- **RbtAct** ([2603.09723](https://arxiv.org/abs/2603.09723)) — built `REVIEWPREF-DPO-22K`,
  21,822 same-paper style-matched ICLR review pairs arbitrated by the **author**. The
  nearest neighbour; same-paper matching is theirs, not ours.
- **Score incomparability across areas** ([2607.27209](https://arxiv.org/abs/2607.27209)) — H3's premise.
- **BridgeAlign** ([2607.27366](https://arxiv.org/abs/2607.27366)) — the style-over-substance failure mode.
- **PLURAL** ([2607.08034](https://arxiv.org/abs/2607.08034)) — DPO compresses group-specific variation.
- **Likelihood displacement** ([2410.08847](https://arxiv.org/abs/2410.08847)) — the mechanism behind the diagnostics.
- **SNOR v1** (DOI [10.5281/zenodo.15866613](https://zenodo.org/records/15866613), CC BY 4.0) — the review corpus.
- **papercopilot/paperlists** — ratings and outcomes used for the yield validation.
- **DiscoPOP** ([2406.08414](https://arxiv.org/abs/2406.08414)) and **ShinkaEvolve** — the lineage of the trainer.

---

MIT license (code), with the use restriction stated in Ethics above. Review text remains
CC BY 4.0 and is not redistributed here. Where MIT and that restriction conflict, we ask
users to honour the restriction.
