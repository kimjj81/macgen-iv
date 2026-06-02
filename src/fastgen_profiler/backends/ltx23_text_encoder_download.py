"""Download helper for LTX2.3 text encoder assets."""
from __future__ import annotations

from pathlib import Path


_MAX_TEXT_ENCODER_SCAN_FILES = 10_000


def ensure_text_encoder(
    dest_dir: Path,
    auto_download: bool = False,
) -> tuple[Path, Path]:
    """Ensure Gemma3 text encoder + tokenizer are available.

    Returns (text_encoder_dir, tokenizer_dir).

    Args:
        dest_dir: Base directory where ``text_encoder/`` and ``tokenizer/``
                  will be placed.  Defaults to
                  ``<model_path>/../LTX-2-text-local``.
        auto_download: If True, download missing files automatically.
                       If False and files are missing, raise FileNotFoundError.
    """
    text_encoder_dir = dest_dir / "text_encoder"
    tokenizer_dir = dest_dir / "tokenizer"

    if _is_text_encoder_ready(text_encoder_dir, tokenizer_dir):
        return text_encoder_dir, tokenizer_dir

    if not auto_download:
        _raise_missing(text_encoder_dir, tokenizer_dir)

    _preflight_auto_download(dest_dir)

    # ── Auto-download ──────────────────────────────────────────────
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install huggingface_hub"
        )

    hf_repo = "google/gemma-3-12b-it"
    print(f"[macgen-iv] Downloading text encoder from {hf_repo} ...")
    print(f"[macgen-iv] Destination: {dest_dir}")
    print(f"[macgen-iv] This may take a while (model is ~25 GB in bf16).")

    # Download only the files we need:
    #   - config.json (top-level, for text_config)
    #   - generation_config.json
    #   - tokenizer files
    #   - model shards (safetensors)
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(text_encoder_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "*.model",          # sentencepiece tokenizer
            "tokenizer*.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model*.safetensors",
        ],
    )

    # Tokenizer files were downloaded into text_encoder_dir.
    # Move/create a symlink so tokenizer_dir points to the same location
    # (AutoTokenizer can load from the text_encoder dir too).
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for f in _bounded_files(text_encoder_dir, "link tokenizer files"):
        if _is_tokenizer_file(f.name) and not (tokenizer_dir / f.name).exists():
            (tokenizer_dir / f.name).symlink_to(f)

    if not _is_text_encoder_ready(text_encoder_dir, tokenizer_dir):
        _raise_missing(text_encoder_dir, tokenizer_dir)

    print("[macgen-iv] Text encoder download complete.")
    return text_encoder_dir, tokenizer_dir


def _preflight_auto_download(dest_dir: Path) -> None:
    from fastgen_profiler.mlx_guard import check_memory_guard

    check_memory_guard(label="ltx2.3 text_encoder auto-download")
    if dest_dir.exists():
        for _ in _bounded_files(dest_dir, "preflight auto-download destination"):
            pass


def _is_text_encoder_ready(te_dir: Path, tok_dir: Path) -> bool:
    if not te_dir.exists() or not tok_dir.exists():
        return False
    has_config = (te_dir / "config.json").exists()
    has_tokenizer = any(_is_tokenizer_file(f.name) for f in _bounded_files(tok_dir, "scan tokenizer"))
    has_weights = any(f.name.endswith(".safetensors") for f in _bounded_files(te_dir, "scan text_encoder"))
    return has_config and has_tokenizer and has_weights


def _is_tokenizer_file(name: str) -> bool:
    return name in (
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ) or name.endswith(".model")


def _raise_missing(te_dir: Path, tok_dir: Path) -> None:
    te_status = "exists" if te_dir.exists() else "MISSING"
    tok_status = "exists" if tok_dir.exists() else "MISSING"
    raise FileNotFoundError(
        f"LTX2.3 text encoder not found.\n"
        f"  text_encoder_dir: {te_dir} ({te_status})\n"
        f"  tokenizer_dir:    {tok_dir} ({tok_status})\n"
        f"\n"
        f"To fix, either:\n"
        f"  1. Set auto_download=True explicitly to download from HuggingFace.\n"
        f"  2. Provide text_encoder_dir/tokenizer_dir explicitly.\n"
        f"  3. Place them at <model_path>/../LTX-2-text-local/."
    )


def _bounded_files(path: Path, label: str):
    scanned = 0
    try:
        for entry in path.iterdir():
            scanned += 1
            if scanned > _MAX_TEXT_ENCODER_SCAN_FILES:
                _raise_scan_error(
                    f"LTX2.3 text encoder {label} exceeded {_MAX_TEXT_ENCODER_SCAN_FILES} files in {path}; "
                    "refusing unbounded local asset scan"
                )
            if entry.is_file():
                yield entry
    except OSError:
        _raise_scan_error(
            f"LTX2.3 text encoder {label} could not scan {path}; refusing unbounded local asset scan"
        )


def _raise_scan_error(message: str) -> None:
    from fastgen_profiler.mlx_guard import MemoryGuardError

    raise MemoryGuardError(message)
