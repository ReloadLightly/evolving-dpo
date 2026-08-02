# Running the Phase-0 baseline on free Colab

1. colab.research.google.com → New notebook → Runtime → Change runtime type →
   **T4 GPU**.

2. Cell 1 — clone and install (torch is preinstalled on Colab):

   ```
   !git clone https://github.com/ReloadLightly/evolving-dpo.git
   %cd evolving-dpo
   !pip -q install transformers datasets peft accelerate
   ```

3. Cell 2 — plumbing check (~1 min, tiny model, synthetic data):

   ```
   !python scripts/run_baseline.py --dry-run
   ```

4. Cell 3 — the real thing (~30–45 min: Qwen3-0.6B, 2k pairs, 300 steps):

   ```
   !python scripts/run_baseline.py --loss dpo
   ```

   Watch the log: `loss` should drift down from ~0.693 (ln 2 — recognize it?),
   `acc` up from ~0.5. Those two numbers moving is DPO working.

5. Cell 4 — download the results file before the runtime dies:

   ```
   from google.colab import files
   files.download("results/baseline_Qwen3-0.6B.json")
   ```

Free-tier survival notes: T4 sessions can disconnect after ~90 idle minutes —
keep the tab focused during the run. If you hit OOM, retry with
`--n-train 1000` (shorter run) or edit `TrainConfig.max_completion_tokens`
down to 256. Phase-1 grid (`--grid`) is ~4× the time of one baseline; run it
as four separate sessions if needed.
