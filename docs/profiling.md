# Profiling

Profiling must measure both phase-level and step-level behavior. Timing boundaries should include explicit MLX synchronization where asynchronous work could otherwise make timings misleading.

## Phase-Level Timing

Every benchmark run should record these phases when applicable:

- `model_load`
- `tokenizer_or_prompt_prepare`
- `text_encoder`
- `latent_init`
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

Benchmark results are written to JSONL. Each line should represent one complete benchmark run with:

- run configuration
- environment summary
- phase timings
- step timings
- quality and speed settings
- generated artifact paths
- notes about skipped or approximate measurements
