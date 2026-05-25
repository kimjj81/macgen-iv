# Model Targets

Wan2.2 and LTX2.3 are the initial target model families.

The implementation should use backend adapter interfaces so exact model loading logic can be replaced later. Model-specific setup belongs in adapters, not in the profiler core.

## Initial Targets

### Wan2.2

The Wan2.2 adapter should expose the common profiler phases while allowing the underlying MLX model loading and pipeline implementation to change.

### LTX2.3

The LTX2.3 adapter should follow the same adapter contract as Wan2.2 so benchmark records remain comparable across model families.

## Core Constraint

Do not hard-code one model pipeline into the profiler core. The profiler coordinates timing, synchronization, metrics, and reporting through common interfaces.

## Local Model Discovery

Model files are discovered from local directories only. The profiler must not download model weights.

Directory sources:

- `--model-dir PATH`, repeatable
- `--model-path PATH`, direct model directory
- `fastgen-profile models import --source SOURCE`
- `.env` key `FASTGEN_MODEL_DIRS`
- `.env` key `FASTGEN_MODEL_DIR_WAN22`
- `.env` key `FASTGEN_MODEL_DIR_LTX23`

Import sources:

- `drawthings`
- `comfyui`
- `huggingface`
- `lmstudio`
- `ollama`
- `all`

Discovery markers:

- `model_index.json`
- `config.json`
- `*.safetensors`
- `*.ckpt`
- `*.gguf`
- `*.mlx`

The discovery layer records selected `model_id`, `model_path`, and `model_source_root` in JSONL output. Family matching is best-effort from path text; unknown candidates can still be listed for operator inspection.

Ollama directories can be registered so their locations are visible, but direct Ollama blob/manifests loading is not implemented by the profiler yet.

Import replaces `FASTGEN_MODEL_DIRS` with the directories found in the current scan. Existing unrelated `.env` keys and comments are preserved, but stale model directory entries are removed. If no known model directories exist, import exits with an error and leaves `.env` unchanged.
