# HANDOFF — Institutional Preference Pairs from Peer Review

**Version 1.1** — For: Claude Code, working in Roland's repo.
Written 2026-08-14 by Claude (Cowork session). Everything marked VERIFIED was
actually executed and observed, not inferred. Everything marked UNVERIFIED needs
checking on Roland's machine before you build on it.

---

## 0. Read this first — what to do in your first session

1. Read §2 (the question) and §3 (verified state). Do not re-derive them.
2. Do **not** re-run the literature search. §7 has it, with arXiv IDs.
3. Start at **Task 2** in §5. Task 1 is already done — the numbers are in §3.
4. **§12–§16 are the publication plan: hard deadline, claim scope, reviewer
   objections, figures. Read them before writing any analysis code — they
   determine what has to be measured.**
5. The one thing only Roland can supply is the sentence in §2.1. If it is still
   blank, proceed anyway; it blocks the paper's framing, not the code.

**The single most important environment fact:** the Cowork sandbox that wrote
this document cannot reach `api.openreview.net` or `huggingface.co` (both 403 at
an egress proxy). Roland's machine can. That asymmetry is why this handoff
exists. If you are running on Roland's machine, you have access the previous
session did not — check before assuming a blocker.

---

## 1. What this project is, in three sentences

Peer review is an institution that makes recorded decisions about competing
judgements. When two reviewers of the same paper disagree, the institution
eventually sides with one of them. We turn that into preference-optimization
training data and ask whether a model tuned on it learns *what a research
community counts as a good argument* — or merely *how its reviewers write*.

---

## 2. The research question and hypotheses

### 2.1 The one-sentence question

Claude's placeholder, **to be replaced by Roland's own wording**:

> When you preference-tune a language model on real peer-review decisions, does
> it learn what a research community counts as a good argument — or only how its
> reviewers write?

**Roland's version:** _______________________________________

### 2.2 Hypotheses (pre-registered; written before any model was trained)

**H1 — Learnability.** A model preference-tuned on institutionally-arbitrated
review pairs identifies, on held-out papers, which of two divergent reviews the
institution followed, better than (i) the untuned base model, (ii) an
SFT-on-chosen-only control, and (iii) a surface-feature classifier.
*Falsified if* it fails to beat all three with non-overlapping 95% CIs over 5 seeds.

**H2 — Substance over style.** The H1 advantage survives length- and
sentiment-matching, and collapses on shuffled pairs (reviews from different papers).
*Falsified if* the advantage vanishes under matching, or persists under shuffling.

**H3 — Community specificity. This is the contribution.** Let A and B be two
research areas with documented different acceptance thresholds. A model tuned on
A scores higher on A's held-out pairs than on B's, and vice versa — i.e. a
significant `tuning_area × test_area` interaction.
*Falsified if* the interaction is not significant. That null is informative: it
would mean DPO cannot capture a difference the decision statistics show is present.

**H4 — Optimizer sensitivity (exploratory, only if time allows).** The size of the
H3 interaction differs between DPO and a reference-free length-normalized
objective (SimPO). No confirmatory claim will be made from it.

### 2.3 Why H3 is not a guess

"Reviewer Scores Are Not Comparable Across Research Areas"
([arXiv:2607.27209](https://arxiv.org/abs/2607.27209)) — ICLR 2021–2026, 50,289
papers, 219 topics: at an **identical** reviewer score, acceptance probability
varies up to **8×** by topic; thresholds span 0.812 rating points; topic identity
adds explanatory power beyond score (χ²=341.2, p=5.1e-3); scoring culture explains
only 21% of the gap. The differences exist. The question is whether DPO captures them.

---

## 3. Verified state — measured, not estimated

### 3.1 Data yield (VERIFIED — this was actually computed)

Source: `github.com/papercopilot/paperlists`, files `iclr/iclr20XX.json`.
Rating field is `rating` for 2020, 2021, 2024 and `recommendation` for 2022, 2023.
"Split" = max reviewer rating − min reviewer rating on the same paper.

| Year | Papers | With ≥2 ratings | Spread ≥3 | rate | Spread ≥4 | rate | Non-withdrawn |
|---|---|---|---|---|---|---|---|
| 2020 | 2,594 | 2,561 | 1,225 | 47.8% | 603 | 23.5% | 2,213 |
| 2021 | 3,014 | 2,979 | 1,001 | 33.6% | 379 | 12.7% | 2,616 |
| 2022 | 3,422 | 3,375 | 1,572 | 46.6% | 512 | 15.2% | 2,677 |
| 2023 | 4,955 | 4,919 | 2,314 | 47.0% | 818 | 16.6% | 3,860 |
| 2024 | 7,407 | 7,261 | 3,474 | 47.8% | 1,159 | 16.0% | 5,698 |
| **Total** | **21,392** | **21,095** | **9,586** | **45.4%** | **3,471** | **16.5%** | **17,064** |

**ICLR 2025 and 2026 are NOT in this count** — those files are git-LFS pointers
and the LFS fetch was blocked by the sandbox proxy. On Roland's machine,
`git lfs pull` should retrieve them, adding roughly 11.5k + 19k more submissions.

**Consequence: the preregistration's estimate of 8,000–12,000 pairs is confirmed
at the low end and will roughly double once 2025–2026 are included. The kill
criterion (abandon below 3,000 pairs) is cleared with large margin, at both the
≥3 and the stricter ≥4 spread threshold.**

### 3.2 What papercopilot does NOT contain

Per-reviewer **rating values**, yes. Review **text**, no — only word counts
(`wc_review`, `wc_main_review`). So the text still has to come from OpenReview.
That is Task 2.

### 3.3 The existing code

`github.com/ReloadLightly/evolving-dpo` — VERIFIED working. Cloned to a clean
machine, `pytest -q` gave **15 passed, 3 skipped** under torch 2.13 /
transformers 5.15 / peft 0.20. Reusable pieces:

- `src/evolving_dpo/trainer.py` — LoRA DPO trainer. Reference model = same model
  with adapters disabled, so no second model in memory. Skips reference passes
  entirely for losses marked `needs_reference = False`.
- `src/evolving_dpo/losses.py` — dpo, ipo, slic, dpop, simpo; uniform signature.
- `src/evolving_dpo/evaluate.py` — `score()` already computes `pref_accuracy`,
  which **is** this project's primary metric: does the model's implicit reward
  rank the chosen item above the rejected one. Also computes per-token drift and
  `chosen_logp_shift` (likelihood displacement).
- `src/evolving_dpo/data.py` — `encode_pair`/`collate`; needs a new loader for
  review pairs, everything else is reusable.

The repo has **never been run on real data.** No numbers exist in it.

---

## 4. Environment notes

| Thing | Cowork sandbox | Roland's machine (assumed) |
|---|---|---|
| `git clone` public GitHub | ✅ VERIFIED | ✅ |
| `git lfs pull` | ❌ proxy 403 | UNVERIFIED — try it |
| `api.openreview.net` | ❌ proxy 403 | UNVERIFIED — try it |
| `huggingface.co` | ❌ proxy 403 | UNVERIFIED — try it |
| PyPI (`pip install`) | ✅ VERIFIED | ✅ |
| `git push` | ❌ proxy blocks credential injection | ✅ |
| GPU | none | none — use Kaggle / Colab / rented |

**Compute plan (researched 2026-08-14):** do not buy hardware. Kaggle gives 30
GPU-hours/week free on a P100 (16 GB); Colab free gives a T4 (15 GB). Both run
QLoRA and DPO for 0.6B–8B. QLoRA needs ~1.5 GB at 0.6B and ~11–13 GB at 7–8B with
seq 2048; DPO needs ~1.5–2× the SFT figure because it processes chosen and
rejected together. Rented RTX 4090 (24 GB) on Vast.ai is ~$0.39/h if more is needed.

---

## 5. Task list

Each task has an exit condition. Do not start the next before the exit condition
holds. Commit after each.

### Task 1 — Measure pair yield ✅ DONE
See §3.1. Optionally extend: `git lfs pull` the 2025/2026 files and recompute.
*Exit: table in §3.1 extended to 2026.*

### Task 2 — Fetch review text from OpenReview
Build `scripts/fetch_openreview.py`.

- Client: `pip install openreview-py` (2.4.2 was current). ICLR ≤2023 uses
  `openreview.Client(baseurl="https://api.openreview.net")` (v1); ICLR 2024+ uses
  `openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")` (v2).
  Anonymous access to public venues is expected to work — **verify first**.
- v1: `get_all_notes(invitation=f"ICLR.cc/{year}/Conference/-/Blind_Submission",
  details="directReplies")` gets submissions and their replies in one call.
  Fall back to `.../-/Submission` if that returns nothing.
- v2: `get_all_notes(invitation=f"ICLR.cc/{year}/Conference/-/Submission")` then
  `get_all_notes(forum=sub.forum)` per paper. Reviews are replies whose
  `invitations` contain `Official_Review`.
- Content shape differs: v1 `content = {k: v}`, v2 `content = {k: {"value": v}}`.
  Normalize before touching anything.
- **Cache raw JSON to disk per year before parsing.** The download is the
  expensive part; you will want to re-parse many times without re-downloading.
- Field names differ by year — 2022/2023 use `recommendation`, `correctness`,
  `technical_novelty`, `empirical_novelty`; 2020/2021/2024 use `rating`,
  and 2024+ adds `soundness`, `presentation`, `contribution`. Detect, don't assume.

*Exit: `data/raw/iclr{year}.jsonl` exists for 2020–2025, with per-paper review
text and per-review ratings, and a printed count matching §3.1 within ~2%.*

### Task 3 — Construct the pairs
Build `scripts/build_pairs.py` producing `data/pairs.jsonl` with this schema:

```json
{"paper_id": "...", "year": 2024, "primary_area": "...",
 "prompt": "<title>\n\n<abstract>",
 "chosen": "<full text of the review on the side of the final decision>",
 "rejected": "<full text of the review on the losing side>",
 "chosen_rating": 8, "rejected_rating": 3,
 "decision": "Poster", "accepted": true,
 "majority_agreed_with_decision": true,
 "chosen_len_tokens": 812, "rejected_len_tokens": 655}
```

Rules:
- Include a paper only if reviewer ratings straddle the venue's accept threshold
  (use the empirical threshold per year, not a hardcoded 5).
- `chosen` = the review whose recommendation matched the final decision.
- Exclude withdrawn and desk-rejected papers (`status` contains "Withdraw"/"Desk").
- **Exclude ICLR 2026 entirely** — organizers reset all scores to a pre-rebuttal
  state mid-discussion after a security incident, so its record is not a clean
  institutional trace.
- Record `majority_agreed_with_decision`; §6 needs it.
- If several reviews sit on each side, take the most extreme pair, and record
  how many alternatives were discarded.

*Exit: `data/pairs.jsonl` with ≥3,000 rows (expect ~9,000+), plus a
`data/DATASET_CARD.md` reporting counts by year and area, the licence
(OpenReview comments are CC BY 4.0), and the exclusions above.*

### Task 4 — Validate the harness before trusting it
Reproduce a published DPO preference-accuracy figure on UltraFeedback with the
existing `evolving-dpo` trainer at Qwen3-0.6B.

*Exit: a number in the README, with the source it is being compared against.
If this fails, stop — kill criterion 3. Nothing downstream is trustworthy.*

### Task 5 — Train the five arms
`base`, `sft` (chosen only), `dpo`, `dpo-shuffled`, `surface`.
Qwen3-0.6B + LoRA r=16. **5 seeds each.** Tune β for the `dpo` arm on a
validation split — the treatment does not get an advantage the baseline was denied.

Splits are structural, never random:
- **Temporal**: train 2020–2023, test 2024–2025.
- **Topic**: for H3, split by `primary_area`; choose the two areas *before*
  training and record the choice.

*Exit: `results/main.csv` with mean and 95% CI per arm.*

### Task 6 — The H3 crossover
Two areas × two tuned models × both test sets. Fit a logistic regression on
pair-level correctness with `tuning_area`, `test_area`, and their interaction.
**The interaction coefficient is the result.** A main effect alone does not
support H3.

*Exit: `results/crossover.csv` plus the fitted interaction term with CI.*

### Task 7 — Write-up
4–8 pages. arXiv preprint plus a workshop. Do not claim main-track scope.

---

## 6. The traps — read before writing any evaluation code

**The 70% trap (this one kills careless versions of the study).** About 70% of
ICLR submissions are rejected, and reviews recommending rejection read
differently from reviews recommending acceptance. A model that always prefers the
more negative review scores ~70% and looks like a result.
**Mitigation, non-negotiable: balance the evaluation set 50/50 between accepted
and rejected papers**, so that strategy scores 50%. Report the always-negative
baseline in every table.

**Agreement is not correctness.** A review matching the decision may have matched
it by predicting the majority. Report accuracy separately on the subset where the
decision went *against* the numerical majority — that is the honest test, and it
is why Task 3 records `majority_agreed_with_decision`.

**Style is the default thing to learn.** BridgeAlign
([arXiv:2607.27366](https://arxiv.org/abs/2607.27366)) tested provenance-based
pairs (human = chosen, model = rejected) and it **underperformed the untuned base
model**, 60.40 vs 60.70, because the model exploited source cues. Our pairs are
same-paper same-genre, which is the defence — but the surface-feature classifier
baseline is what actually proves it. If the LM does not beat logistic regression
on review length, question count, hedge words, sentiment and citation counts,
we learned style. Report it either way.

**DPO may compress what we are measuring.** PLURAL
([arXiv:2607.08034](https://arxiv.org/abs/2607.08034)) found DPO retains ~18% of
cross-country value variation where SFT retains 30%. If our H3 interaction is
weak, this is a candidate explanation and the SFT arm is the comparison that
tests it.

**Contamination.** One estimate puts ~21% of recent ICLR reviews as
AI-generated. Unfixable for 2024–2025. Run H1 on 2020–2022 only as a sensitivity
check and report both.

---

## 7. Literature — do not re-search this

**The paper that owns the mechanism.** RbtAct,
[arXiv:2603.09723](https://arxiv.org/abs/2603.09723) (Mar 2026) — built
`REVIEWPREF-DPO-22K`: 21,822 chosen/rejected pairs of real ICLR 2024 review text,
**constrained to the same paper and same perspective to control for confounds**,
then SFT + DPO on Llama-3.1-8B. Their arbiter is the **author** (did the rebuttal
address the point). Ours is the **institution**. Cite them early and explicitly;
do not claim style-matching as novel.

**What remains open (verified by adversarial search, nothing found):** preference
pairs arbitrated by the institution's own verdict; a cross-community transfer
experiment with review models; likelihood-displacement measurement in this setting.

**Essential background:**
- Score incomparability across areas: [2607.27209](https://arxiv.org/abs/2607.27209) — the premise for H3.
- Corpus landscape: Re², [2505.07920](https://arxiv.org/abs/2505.07920).
- Style-vs-substance failure: BridgeAlign, [2607.27366](https://arxiv.org/abs/2607.27366).
- DPO compresses variation: PLURAL, [2607.08034](https://arxiv.org/abs/2607.08034).
- LLMs vs human reviews: Liang et al., [2310.01783](https://arxiv.org/abs/2310.01783).
- Rebuttal score dynamics: [2606.22166](https://arxiv.org/abs/2606.22166), [2511.15462](https://arxiv.org/abs/2511.15462).
- UNBench (the abandoned UN-votes substrate): [2502.14122](https://arxiv.org/abs/2502.14122).
- Older corpora: PeerRead [1804.09635](https://arxiv.org/abs/1804.09635), NLPeer [2211.06651](https://arxiv.org/abs/2211.06651), DISAPERE [2110.08520](https://arxiv.org/abs/2110.08520), ARIES [2306.12587](https://arxiv.org/abs/2306.12587), MReD [2110.07474](https://arxiv.org/abs/2110.07474).

**Ruled out, with reasons:**
- *Pre/post-rebuttal pairs* — OpenReview does not publicly expose review edit
  history. Only `papercopilot/iclr-insights` has initial vs final scores (2024–25,
  CC BY 4.0), and it has scores, not pre-rebuttal text.
- *Meta-review-sided-with labels* — meta-reviews almost never name reviewers, so
  the label needs LLM adjudication, importing judge noise into ground truth.
  Keep as a v2 refinement.
- *NeurIPS / ICML / ARR* — NeurIPS publishes reviews only for accepted papers;
  ICML does not publish reviews; ARR reviews are closed. ICLR is the only usable venue.

---

## 8. Ethics — must appear in the paper

Reviews on OpenReview are public and CC BY 4.0, and RbtAct and Re² establish
precedent. But reviewers wrote them for authors and area chairs, not as
preference-optimization data. Commit to:

- releasing pair **identifiers and construction code**, not a bulk redistribution
  of review text, so the licence chain stays intact;
- no reviewer de-anonymization and no per-reviewer analysis;
- an explicit statement that the artifact must not be used in a deployed
  reviewing system — cf. "Stop Automating Peer Review Without Rigorous
  Evaluation", [arXiv:2605.03202](https://arxiv.org/abs/2605.03202).

---

## 9. Reading list for Roland, in order

While Claude Code does Tasks 2–3, these are the four that change what you think:

1. **RbtAct** [2603.09723](https://arxiv.org/abs/2603.09723) — read first. It is
   the nearest neighbour and you need to be able to say in one sentence how your
   design differs. (Answer: their arbiter is the author, yours is the institution.)
2. **Score incomparability** [2607.27209](https://arxiv.org/abs/2607.27209) —
   the empirical foundation of H3. Read the topic-effect tables closely.
3. **BridgeAlign** [2607.27366](https://arxiv.org/abs/2607.27366) — read §H-MPO.
   This is the failure mode the whole design is built to avoid.
4. **Razin et al., likelihood displacement**
   [2410.08847](https://arxiv.org/abs/2410.08847) — the mechanism behind the
   diagnostics already in `evaluate.py`, and the bridge back to the
   optimizer question if H4 ever gets run.

Optional, for the neuroevolution thread you started from: DiscoPOP
[2406.08414](https://arxiv.org/abs/2406.08414), ShinkaEvolve
[2509.19349](https://arxiv.org/abs/2509.19349), ES at Scale
[2509.24372](https://arxiv.org/abs/2509.24372), and the ES-geometry rebuttal
[2604.01499](https://arxiv.org/abs/2604.01499).

---

## 10. Open decisions — Roland's, not Claude Code's

1. **The sentence in §2.1.** If it is not in his words, the project is not his.
2. **Which two areas for H3.** Maximum documented threshold separation is one
   criterion; his own judgement about which two communities genuinely differ is
   another, and probably better.
3. **Repo.** Default suggestion: a new repo, because the science is no longer
   "evolving DPO" — copy `src/evolving_dpo/` in as a package. A branch of
   `evolving-dpo` also works and breaks nothing.
4. **Whether H4 stays.** Dropping it makes the paper cleaner. Keeping it
   preserves the link to the neuroevolution thread the project started from.

---

## 11. Files that came out of the Cowork session

- `preregistration-v0.1.md` — hypotheses, arms, metrics, kill criteria, threats
  to validity. This handoff is its operational version; the prereg is the one
  that gets timestamped in git.
- `dpo-x-neuroevolution-map.md` — 23-paper map of where evolution and
  post-training meet. Background for the H4 thread; not needed for Tasks 2–7.
- `probe_openreview.py` — structure probe for the OpenReview API. Superseded by
  §3.1 for the yield question, but still useful in Task 2 to discover per-year
  field names before writing the extractor.

---

# PUBLICATION PLAN

Added in v1.1. §1–§11 describe a sound study. §12–§16 are what turns a sound
study into a submitted paper. They are not optional.

## 12. Target venues, with real deadlines

Verified against official pages on 2026-08-15.

| Deadline | Venue | Format | Archival | Assessment |
|---|---|---|---|---|
| **29 Aug 2026** AoE | [AI for Meta-Science @ NeurIPS 2026](https://ai4metascience.org/) | 4 pp technical | **No** | Exact topical match — scope is quality control and evaluation of science under AI. Non-archival, so submitting **cannot** block ARR. |
| **29 Aug 2026** AoE | [AI-Native Academia @ NeurIPS 2026](https://ai-native-academia.github.io/) | 4 pp short | Double-blind | Explicit track on AI-assisted peer review and reviewer accountability. |
| **12 Oct 2026** | [ARR October cycle](https://aclrollingreview.org/dates) → NAACL 2027 / COLING 2027 (commit by 20 Dec) | ~4 pp short | **Yes** | **Primary target.** Archival, short-paper track, matches the realistic timeline. |
| Rolling | [TMLR](https://jmlr.org/tmlr/faq.html) | any length | Yes | Safety net. ~9 weeks to decision. Judges claim-support rigour rather than novelty, which suits this study. |

**Primary: ARR, 12 October 2026.** **Backup: TMLR, no deadline.**
**Optional sprint: AI for Meta-Science, 29 August 2026** — see §12.1.

Not viable: ICLR 2027 main track (25 Sep, full-length, wrong scope);
NeurIPS 2026 main track (closed May); EMNLP 2026 workshops (none on peer review).

### 12.1 The schedule constraint that actually governs this project

**Roland's Claude subscription ends 31 August 2026.** After that, Claude Code is
not available. This is the binding constraint, not the ARR deadline.

Therefore the plan is split at 31 August:

**Before 31 Aug — everything that needs Claude Code.** All code written, data
built, experiments runnable by a single command, analysis scripts written, paper
skeleton with section headings and the figure-generating code in place.

**After 31 Aug — everything Roland can do alone.** Running the experiments on
Kaggle, reading the outputs, writing prose, submitting.

Concretely, by 31 August the repo must contain:

```
data/pairs.jsonl                 # built, validated, with dataset card
scripts/fetch_openreview.py      # done
scripts/build_pairs.py           # done
scripts/run_all.py               # ONE command runs all arms x 5 seeds
scripts/analyze.py               # reads results/, emits every table and figure
paper/skeleton.md                # sections, claims, figure captions written
paper/figures/                   # generated by analyze.py, empty for now
```

If it is 29 August and the experiments have not run, **that is fine** — what must
not be missing is the code that runs them.

**The sprint option:** a 4-page version to AI for Meta-Science by 29 August is
non-archival, so it costs nothing if rejected and does not block ARR. It also
forces the whole artifact to exist while Claude Code is still available. Only
attempt it if `data/pairs.jsonl` exists by 22 August.

## 13. Additional tasks (extend §5)

### Task 8 — Power analysis, BEFORE training
Given the pair count from Task 3 and 5 seeds per arm, compute the minimum
detectable difference in pairwise accuracy at 80% power, α = 0.05, for (a) the
H1 comparisons and (b) the H3 interaction term. Interactions need substantially
more data than main effects; if the H3 test is underpowered at the planned area
split, find that out now and merge areas or increase the pair count instead of
discovering it in the results.

*Exit: `results/power.md` stating the minimum detectable effect for each
hypothesis, and a decision on the area split that respects it.*

### Task 9 — Human validation of the label, 100 pairs
Roland personally reads 100 randomly sampled pairs, blind to the label, and
records which review he would have sided with and why in one line.

This is the answer to the reviewer question *"is your automatic label
meaningful?"*, and there is no substitute for it. Report agreement between
Roland's judgement and the institutional label. **A low agreement rate is a
finding, not a failure** — it would mean the institution's decisions do not track
what a trained reader sees as the better review, which is itself publishable.

*Exit: `data/human_validation.csv` (100 rows) and an agreement rate with CI.*

### Task 10 — Scale check at 7–8B
Rerun the winning configuration at Qwen3-8B or Llama-3.1-8B with QLoRA on
Kaggle's free P100 (16 GB, 30 h/week). QLoRA at 8B needs ~11–13 GB at seq 2048;
DPO needs ~1.5–2× the SFT figure, so keep sequence length and batch modest.

Without this, the strongest reviewer objection is "0.6B tells us nothing."
With it, the paper reports the effect at two scales.

*Exit: `results/scale.csv` with the H1 and H3 numbers at 8B.*

### Task 11 — Reproducibility artifact
Pin dependency versions, fix seeds, script the whole pipeline end to end from an
empty checkout, and write the release note. Release pair **identifiers and
construction code**, not bulk review text (§8).

*Exit: a fresh clone reproduces `results/main.csv` with one command.*

## 14. Claim scope — what this paper does and does not say

Write this into the paper's limitations section verbatim. Overclaiming is the
most common cause of rejection for short empirical papers, and this study has
four honest limits.

**We claim:** that institutionally-arbitrated review pairs are learnable by
preference optimization above named baselines (H1); that the learned signal is
not reducible to surface style (H2); and that it is or is not community-specific
(H3, either direction).

**We do not claim:** that the model judges review quality; that it should be used
in any reviewing system; that the institution's decision was correct; or that
anything here generalizes beyond ICLR, which is one venue in one field.

**Named limitations, stated by us before a reviewer states them:**
1. One venue, one field, English only.
2. Model scale 0.6B, with an 8B check (Task 10) rather than a scaling study.
3. Agreement with the outcome is not correctness — see the against-the-majority
   subset analysis in §6.
4. An estimated ~21% of recent ICLR reviews may be AI-generated; the 2020–2022
   sensitivity analysis is the response, not a fix.

## 15. Reviewer objections, and where each is answered

Anticipate these five. If any has no answer in the results, the paper is not
ready.

| Objection | Where it is answered |
|---|---|
| "This is just RbtAct." | §7. Their arbiter is the author; ours is the institution. State this in the abstract, not the related-work section. |
| "The model learned to pick negative reviews." | §6. The evaluation set is balanced 50/50, and the always-negative baseline appears in every table. |
| "It learned writing style, not judgement." | H2, plus the surface-feature classifier baseline, plus the shuffled-pairs arm. Three independent checks, not one. |
| "0.6B is a toy." | Task 10 — the 8B rerun. |
| "Your labels are automatic and meaningless." | Task 9 — 100 human-validated pairs with an agreement rate. |

## 16. The three figures

A short paper is remembered by its figures. Write `analyze.py` to emit exactly
these, and design the experiments so they can be filled.

**Figure 1 — the dataset.** Pair count by year and by research area, with the
rating-spread distribution. Establishes that the data is real and non-trivial.

**Figure 2 — H1 and H2 together.** Pairwise accuracy by arm, with 95% CIs over
5 seeds, and the two baselines (always-negative at 50%, surface classifier) drawn
as horizontal reference lines. One panel per split (temporal, topic). This is the
main result and it must be readable in grayscale at a glance.

**Figure 3 — the crossover.** A 2×2 of tuning area against test area, accuracy in
each cell, with the interaction coefficient and CI stated in the caption. If the
interaction is null, this figure still carries the paper — draw it either way and
say so in the caption.

Before plotting anything, read the `dataviz` skill if it is available.
