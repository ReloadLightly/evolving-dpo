# evolving-dpo

**Can we (re)discover preference-optimization objectives by evolution — on a
free-tier compute budget?**

Frontier post-training tunes language models with hand-designed preference
losses (DPO and its family). Sakana AI's [DiscoPOP](https://arxiv.org/abs/2406.08414)
(NeurIPS 2024) showed an LLM can *discover* better objectives inside an
evolutionary loop. This project rebuilds that idea as a small, honest, fully
open mini-lab, driven by Sakana's own open-source
[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) framework — and uses
it as a working laboratory for studying RLHF (following
[Lambert's RLHF Book](https://rlhfbook.com/), esp. ch. 8 on direct alignment
and ch. 14 on over-optimization).

The claim is **not** "we beat DPO at scale." The claim is a working, open,
reproducible pipeline for objective discovery at hobby scale, with honest
measurement — including of the failure modes (see *Goodhart watch*, below).

## How it works

```
              ┌────────────────────────────────────────────────┐
              │ losses.py: the plug-in slot                     │
              │   f(policy/ref log-probs, beta) -> loss         │
              │   dpo | ipo | slic | dpop | <evolved candidates>│
              └────────────────┬───────────────────────────────┘
                               │
   trainer.py: minimal DPO-family loop (LoRA; reference model = same
   base weights with adapters disabled — zero extra memory)
                               │
   evaluate.py: fitness = held-out preference accuracy
                          − penalty · max(0, drift − budget)
                               │
   shinka_task/: ShinkaEvolve rewrites the plug-in body, generation by
   generation, maximizing that fitness   (Phase 3)
```

Design choices worth knowing:

- **The loss is data.** Every objective shares one signature (four summed
  log-probs + β in, per-example losses out) — so "an objective" is a short
  function body an LLM can mutate.
- **The reference model is free.** The policy is base-model + LoRA; running
  with adapters disabled *is* the frozen reference. One model in memory.
- **Fitness is guarded.** Preference accuracy alone is trivially hackable
  (win the margin by wrecking the model). A per-token drift term — a cheap
  KL-style proxy — taxes candidates that wander too far from the reference.
- **Divergence is data.** Candidates that NaN out score 0 and report why;
  in Phase 3 that diagnostic is fed back to the mutation LLM as text feedback.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_baseline.py --dry-run     # CPU, ~1 min: plumbing check
pytest -q                                    # unit + smoke tests
```

Real baseline (free Colab T4, ~30–45 min): see
[docs/colab_quickstart.md](docs/colab_quickstart.md).

```bash
python scripts/run_baseline.py --loss dpo    # one loss
python scripts/run_baseline.py --grid        # dpo vs ipo vs slic vs dpop
```

## Roadmap

- [x] **Phase 0 — Baseline.** Minimal DPO trainer + vanilla DPO on
  Qwen3-0.6B / UltraFeedback slice.
- [ ] **Phase 1 — Known variants.** IPO, SLiC, DPOP grid (implemented; run
  the benchmark). Exercise: port DiscoPOP's discovered LRML loss from
  [their repo](https://github.com/SakanaAI/DiscoPOP) and add it to the grid.
- [ ] **Phase 2 — Fitness hardening.** Stress the drift guard; calibrate
  budget/penalty so known-good losses rank sensibly.
- [ ] **Phase 3 — Evolution.** Arm `shinka_task/` and run 30–60 candidates
  on a rented GPU. Archive everything, including the failures.
- [ ] **Phase 4 — Validation & write-up.** Winner re-trained at 4–8B scale
  vs DPO baseline; ablations; report.

## Goodhart watch

A running log (in the eventual report) of every way a candidate exploited the
fitness function rather than solving the task. Reward hacking is
specification gaming — the evolutionary-computation literature documented it
decades before RLHF existed — and a discovery loop like this one is a
specification-gaming generator by construction. Guarding the fitness, watching
candidates probe the guard, and documenting the arms race *is* the experiment.

## Pen-and-paper appendix

The math this repo runs on, in the order it appears in the code:

| Where | Concept |
|---|---|
| `data.py` labels/-100, attention_mask | the three post-training masks |
| `trainer.py::sequence_logps` | chain rule: log P(completion) = Σ token log-probs |
| `losses.py::dpo_loss` | Bradley–Terry via sigmoid; `-logsigmoid(0) = ln 2` |
| β · (policy − ref) log-ratios | implicit reward; the KL leash in disguise |
| `evaluate.py` drift term | KL-style divergence as a regularizing guard |
| `losses.py::ipo_loss` docstring | over-optimization and bounded objectives |

## References

DiscoPOP: [arXiv 2406.08414](https://arxiv.org/abs/2406.08414) ·
ShinkaEvolve: [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) ·
DPO: Rafailov et al. 2023 · IPO: Azar et al. 2023 · SLiC-HF: Zhao et al. 2023 ·
DPOP: Pal et al. 2024 · RLHF Book: [rlhfbook.com](https://rlhfbook.com/) ·
Data: [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) ·
Model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

MIT license.
