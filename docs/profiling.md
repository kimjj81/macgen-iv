# Profiling

Profiling must measure both phase-level and step-level behavior. Timing boundaries should include explicit MLX synchronization where asynchronous work could otherwise make timings misleading.

## Phase-Level Timing

Every benchmark run records these phases:

- `model_load`
- `prompt_prepare`
- `text_encoder`
- `latent_init`
- `denoise_total`
- `denoise_step`
- `vae_decode`
- `video_encode`
- `file_write`
- `total`

If a backend cannot measure a phase separately, it should record the limitation in the run metadata instead of guessing.

## Step-Level Timing

Denoising should support per-step timing so regressions and hot spots can be located inside the generation loop. Each step record should include:

- step index
- duration
- synchronization method
- active sampler
- guidance setting
- cache state
- compile state

## Synchronization

MLX operations can be lazy or asynchronous depending on the operation path. Phase and step timings must synchronize before stopping the timer when needed, for example with `mx.eval(...)` on relevant outputs.

## Output

Benchmark results are written to JSONL. Each line represents one phase or step measurement record. Records are grouped by `run_id`.

Stable record fields:

- `run_id`
- `timestamp_utc`
- `model`
- `backend`
- `model_path`
- `model_id`
- `model_source_root`
- `prompt_hash`
- `negative_prompt_hash`
- `seed`
- `width`
- `height`
- `frames`
- `fps`
- `steps`
- `guidance`
- `quant`
- `cache`
- `compile`
- `phase`
- `step_index`
- `seconds`
- `peak_memory`
- `active_memory`
- `cache_memory`
- `output_path`
- `error`
- `machine`

Prompts are recorded as deterministic SHA-256 hashes rather than raw prompt text.
