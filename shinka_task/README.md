# Phase 3 — the ShinkaEvolve wiring

**Status: prepared, not yet armed.** Nothing here is needed for Phases 0–2.

The contract is already honored by the main package:

- `initial.py` — seed program with the candidate loss inside
  `EVOLVE-BLOCK-START/END` markers and a `run_experiment(**kwargs)` entry
  point, exactly as ShinkaEvolve expects. The seed is vanilla DPO, so
  generation 0 reproduces the Phase-0 baseline.
- `evolving_dpo.evaluate.run_candidate(...)` — returns a metrics dict whose
  `combined_score` is the fitness ShinkaEvolve maximizes (preference accuracy
  minus the drift/Goodhart penalty; divergence → 0).

Still to do when we arm it (against your installed `shinka-evolve` version —
APIs move, trust their `examples/` over this note):

1. `pip install shinka-evolve` (kept out of `requirements.txt` on purpose so
   the Colab baseline install stays light).
2. Write `evaluate.py` here using their `run_shinka_eval()` helper: run
   `run_experiment`, then aggregate to the required dict —
   `{"combined_score": float, "public": {...}, "private": {...}}` (+ optional
   `"text_feedback"` that the mutation LLM gets to read: we'll feed it the
   drift and divergence diagnostics so evolution learns *why* candidates die).
3. Launch small first, e.g. with the Python API:

   ```python
   from shinka.core import ShinkaEvolveRunner, EvolutionConfig
   from shinka.database import DatabaseConfig
   from shinka.launch import LocalJobConfig

   runner = ShinkaEvolveRunner(
       evo_config=EvolutionConfig(init_program_path="shinka_task/initial.py"),
       job_config=LocalJobConfig(eval_program_path="shinka_task/evaluate.py"),
       db_config=DatabaseConfig(),
       max_evaluation_jobs=1,
   )
   runner.run()
   ```

   or the CLI: `shinka_run --task-dir shinka_task --num_generations 10`.
4. Mutation LLM: ShinkaEvolve supports OpenAI/Gemini/local endpoints and
   `headless/<agent>` strings for CLI-agent subscriptions — the latter means
   a Claude Code subscription can drive mutations without per-token API cost.
   Budget note: with ~10–15 min per candidate at 0.6B on a rented 4090/A100,
   a 30–60 candidate run is an afternoon and a few tens of dollars.
