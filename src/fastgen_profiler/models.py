"""Local model directory discovery."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


MODEL_MARKER_FILES = ("model_index.json", "config.json")
MODEL_MARKER_SUFFIXES = (".safetensors", ".ckpt", ".gguf", ".mlx")
TARGET_GENERATION_MODELS = ("wan2.2", "ltx2.3")
IMPORT_SOURCES = ("drawthings", "comfyui", "huggingface", "lmstudio", "ollama", "all")
SKIP_DIR_NAMES = {
    ".cache",
    ".git",
    ".huggingface",
    "__pycache__",
    "node_modules",
}

DEFAULT_IMPORT_PATHS = {
    "drawthings": (
        "~/Library/Containers/Draw Things/Data/Documents/Models",
        "~/Library/Containers/Draw Things/Data",
        "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
        "~/Library/Containers/com.liuliu.draw-things/Data",
        "~/Documents/Draw Things/Models",
    ),
    "comfyui": (
        "~/ComfyUI/models",
        "~/Documents/ComfyUI/models",
    ),
    "huggingface": ("~/.cache/huggingface/hub",),
    "lmstudio": (
        "~/.cache/lm-studio/models",
        "~/Library/Application Support/LM Studio/models",
    ),
    "ollama": ("~/.ollama/models",),
}

DEFAULT_ENV_FILE_MAX_BYTES = 1024 * 1024
DEFAULT_MODEL_DISCOVERY_MAX_DIRS = 100_000
DEFAULT_MODEL_DISCOVERY_MAX_CANDIDATES = 10_000
DEFAULT_MODEL_DISCOVERY_MAX_FILES = 100_000


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    id: str
    name: str
    path: Path
    source_root: Path
    model_family_guess: str
    markers: tuple[str, ...]


def load_env_file(path: Path, *, max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_env_lines(_read_env_lines(path, max_bytes=max_bytes))


def _check_env_file_size(path: Path, *, max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES) -> None:
    if max_bytes <= 0:
        raise ValueError("env file read limit must be positive")
    file_size = path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"{path} exceeds env file read limit: {file_size} bytes > {max_bytes} bytes"
        )


def _read_env_lines(path: Path, *, max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES) -> list[str]:
    _check_env_file_size(path, max_bytes=max_bytes)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            lines.append(raw_line.rstrip("\n\r"))
    return lines


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def model_dirs_from_sources(
    *,
    model: str | None,
    cli_dirs: Iterable[Path] | None,
    env_file: Path,
) -> list[Path]:
    env_values = load_env_file(env_file)
    dirs: list[Path] = []
    dirs.extend(_split_dirs(env_values.get("FASTGEN_MODEL_DIRS")))

    if model == "wan2.2":
        dirs.extend(_split_dirs(env_values.get("FASTGEN_MODEL_DIR_WAN22")))
    elif model == "ltx2.3":
        dirs.extend(_split_dirs(env_values.get("FASTGEN_MODEL_DIR_LTX23")))

    # Auto-discover from well-known sources (DrawThings, HuggingFace cache, etc.)
    dirs.extend(discover_import_dirs("all"))

    if cli_dirs:
        dirs.extend(cli_dirs)

    return _dedupe_paths(dirs)


def discover_import_dirs(source: str) -> list[Path]:
    sources = DEFAULT_IMPORT_PATHS if source == "all" else {source: DEFAULT_IMPORT_PATHS[source]}
    found: list[Path] = []
    for paths in sources.values():
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if path.exists() and path.is_dir():
                found.append(path.resolve())
    return _dedupe_paths(found)


def discover_generation_model_dirs(
    roots: Iterable[Path],
    *,
    max_dirs: int = DEFAULT_MODEL_DISCOVERY_MAX_DIRS,
    max_candidates: int = DEFAULT_MODEL_DISCOVERY_MAX_CANDIDATES,
    max_files: int = DEFAULT_MODEL_DISCOVERY_MAX_FILES,
) -> list[Path]:
    candidates: list[ModelCandidate] = []
    for model in TARGET_GENERATION_MODELS:
        candidates.extend(
            discover_models(
                roots,
                model=model,
                max_dirs=max_dirs,
                max_candidates=max_candidates,
                max_files=max_files,
            )
        )
        _check_candidate_limit(candidates, max_candidates=max_candidates)
    return _dedupe_paths(candidate.path for candidate in candidates)


def merge_model_dirs_into_env(
    env_file: Path,
    new_dirs: Iterable[Path],
    *,
    max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES,
) -> list[Path]:
    env_values = load_env_file(env_file, max_bytes=max_bytes)
    merged = _dedupe_paths(
        [
            *_split_dirs(env_values.get("FASTGEN_MODEL_DIRS")),
            *(Path(path).expanduser().resolve() for path in new_dirs),
        ]
    )
    _write_env_value(
        env_file,
        "FASTGEN_MODEL_DIRS",
        os.pathsep.join(str(path) for path in merged),
        max_bytes=max_bytes,
    )
    return merged


def replace_model_dirs_in_env(
    env_file: Path,
    new_dirs: Iterable[Path],
    *,
    max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES,
) -> list[Path]:
    replacement = _dedupe_paths(Path(path).expanduser().resolve() for path in new_dirs)
    _write_env_value(
        env_file,
        "FASTGEN_MODEL_DIRS",
        os.pathsep.join(str(path) for path in replacement),
        max_bytes=max_bytes,
    )
    return replacement


def discover_models(
    roots: Iterable[Path],
    *,
    model: str | None = None,
    max_dirs: int = DEFAULT_MODEL_DISCOVERY_MAX_DIRS,
    max_candidates: int = DEFAULT_MODEL_DISCOVERY_MAX_CANDIDATES,
    max_files: int = DEFAULT_MODEL_DISCOVERY_MAX_FILES,
) -> list[ModelCandidate]:
    if max_dirs <= 0:
        raise ValueError("model discovery directory limit must be positive")
    if max_candidates <= 0:
        raise ValueError("model discovery candidate limit must be positive")
    if max_files <= 0:
        raise ValueError("model discovery file scan limit must be positive")
    candidates: list[ModelCandidate] = []
    visited_dirs = 0
    for root in _dedupe_paths(roots):
        if not root.exists() or not root.is_dir():
            continue
        for path in _walk_dirs(root):
            visited_dirs += 1
            if visited_dirs > max_dirs:
                raise ValueError(
                    f"model discovery visited more than {max_dirs} directories; "
                    "narrow --model-dir or import source"
                )
            markers = _markers(path, max_files=max_files)
            if not markers:
                continue
            candidate = ModelCandidate(
                id=_candidate_id(root, path),
                name=path.name,
                path=path.resolve(),
                source_root=root.resolve(),
                model_family_guess=guess_model_family(path),
                markers=markers,
            )
            if model is None:
                candidates.append(candidate)
            elif _is_generation_model_candidate(candidate, model=model):
                candidates.append(candidate)
            _check_candidate_limit(candidates, max_candidates=max_candidates)

        # DrawThings-style flat directory: individual model files in a single dir.
        # When a directory contains .ckpt/.safetensors files that match the target
        # model family but the directory itself has no family name, create per-file
        # candidates.
        if model is not None:
            for file_candidate in _discover_flat_model_files(root, model=model, max_files=max_files):
                if file_candidate not in candidates:
                    candidates.append(file_candidate)
                    _check_candidate_limit(candidates, max_candidates=max_candidates)

    return sorted(candidates, key=lambda item: (item.model_family_guess == "unknown", item.id))


def resolve_model_candidate(
    candidates: Iterable[ModelCandidate],
    model_id: str,
) -> ModelCandidate | None:
    for candidate in candidates:
        if model_id in {candidate.id, candidate.name, str(candidate.path)}:
            return candidate
    return None


def direct_model_candidate(path: Path, *, model: str | None = None) -> ModelCandidate:
    resolved = path.resolve()
    return ModelCandidate(
        id=resolved.name,
        name=resolved.name,
        path=resolved,
        source_root=resolved,
        model_family_guess=guess_model_family(resolved) if model is None else model,
        markers=_markers(resolved),
    )


def candidate_to_dict(candidate: ModelCandidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "name": candidate.name,
        "path": str(candidate.path),
        "source_root": str(candidate.source_root),
        "model_family_guess": candidate.model_family_guess,
        "markers": list(candidate.markers),
    }


def guess_model_family(path: Path) -> str:
    text = str(path).lower().replace("_", "-")
    if "wan2.2" in text or "wan-2.2" in text or "wan22" in text or "wan" in text:
        return "wan2.2"
    if "ltx2.3" in text or "ltx-2.3" in text or "ltx23" in text or "ltx" in text:
        return "ltx2.3"
    return "unknown"


def _split_dirs(value: str | None) -> list[Path]:
    if not value:
        return []
    normalized = value.replace(",", os.pathsep)
    return [Path(part).expanduser() for part in normalized.split(os.pathsep) if part.strip()]


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.expanduser())
    return deduped


def _walk_dirs(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        yield current
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                        continue
                    stack.append(Path(entry.path))
        except OSError:
            continue


def _check_candidate_limit(
    candidates: list[ModelCandidate],
    *,
    max_candidates: int,
) -> None:
    if len(candidates) > max_candidates:
        raise ValueError(
            f"model discovery found more than {max_candidates} candidates; "
            "narrow --model-dir or import source"
        )


def _discover_flat_model_files(
    root: Path,
    *,
    model: str,
    max_files: int = DEFAULT_MODEL_DISCOVERY_MAX_FILES,
) -> Iterable[ModelCandidate]:
    """Find individual model files in a flat directory (DrawThings-style).

    When all model weights live as .ckpt/.safetensors/.gguf files in one
    directory (no subdirectories per model), the directory-level discovery
    won't match because the directory name has no model family info.
    Instead, create one candidate per matching file.
    """
    model_suffixes = {".ckpt", ".safetensors", ".gguf", ".mlx"}
    for scanned, entry in enumerate(_safe_iterdir(root), start=1):
        if scanned > max_files:
            raise ValueError(
                f"model discovery scanned more than {max_files} files in {root}; "
                "narrow --model-dir or import source"
            )
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if entry.suffix.lower() not in model_suffixes:
            continue
        family = guess_model_family(entry)
        if family != model:
            continue
        candidate = ModelCandidate(
            id=entry.name,
            name=entry.name,
            path=entry.resolve(),
            source_root=root.resolve(),
            model_family_guess=family,
            markers=(f"*{entry.suffix.lower()}",),
        )
        yield candidate


def _markers(path: Path, *, max_files: int = DEFAULT_MODEL_DISCOVERY_MAX_FILES) -> tuple[str, ...]:
    markers: list[str] = []
    for scanned, child in enumerate(_safe_iterdir(path), start=1):
        if scanned > max_files:
            raise ValueError(
                f"model discovery scanned more than {max_files} files in {path}; "
                "narrow --model-dir or import source"
            )
        try:
            if not child.is_file():
                continue
        except OSError:
            continue
        if child.name in MODEL_MARKER_FILES:
            markers.append(child.name)
        suffix = child.suffix.lower()
        if suffix in MODEL_MARKER_SUFFIXES:
            markers.append(f"*{suffix}")
        if set(MODEL_MARKER_FILES).issubset(markers) and all(
            f"*{suffix}" in markers for suffix in MODEL_MARKER_SUFFIXES
        ):
            break

    return tuple(sorted(set(markers)))


def _safe_iterdir(path: Path) -> Iterable[Path]:
    try:
        iterator = path.iterdir()
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except OSError:
                return
    except OSError:
        return


def _is_generation_model_candidate(candidate: ModelCandidate, *, model: str) -> bool:
    if candidate.model_family_guess != model:
        return False
    if set(candidate.markers) == {"*.gguf"}:
        return False
    if set(candidate.markers) == {"config.json"} and guess_model_family(Path(candidate.name)) != model:
        return False
    return True


def _candidate_id(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_env_value(
    env_file: Path,
    key: str,
    value: str,
    *,
    max_bytes: int = DEFAULT_ENV_FILE_MAX_BYTES,
) -> None:
    line = f"{key}={value}"
    if not env_file.exists():
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(line + "\n", encoding="utf-8")
        return

    lines = _read_env_lines(env_file, max_bytes=max_bytes)
    updated = False
    output: list[str] = []
    for existing in lines:
        stripped = existing.strip()
        prefix = "export " if stripped.startswith("export ") else ""
        compare = stripped[len(prefix) :] if prefix else stripped
        if compare.startswith(f"{key}="):
            output.append(f"{prefix}{line}")
            updated = True
        else:
            output.append(existing)

    if not updated:
        output.append(line)
    env_file.write_text("\n".join(output) + "\n", encoding="utf-8")
