"""Local model directory discovery."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


MODEL_MARKER_FILES = ("model_index.json", "config.json")
MODEL_MARKER_SUFFIXES = (".safetensors", ".ckpt", ".gguf", ".mlx")
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
        "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
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


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    id: str
    name: str
    path: Path
    source_root: Path
    model_family_guess: str
    markers: tuple[str, ...]


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def merge_model_dirs_into_env(env_file: Path, new_dirs: Iterable[Path]) -> list[Path]:
    env_values = load_env_file(env_file)
    merged = _dedupe_paths(
        [
            *_split_dirs(env_values.get("FASTGEN_MODEL_DIRS")),
            *(Path(path).expanduser().resolve() for path in new_dirs),
        ]
    )
    _write_env_value(env_file, "FASTGEN_MODEL_DIRS", os.pathsep.join(str(path) for path in merged))
    return merged


def replace_model_dirs_in_env(env_file: Path, new_dirs: Iterable[Path]) -> list[Path]:
    replacement = _dedupe_paths(Path(path).expanduser().resolve() for path in new_dirs)
    _write_env_value(
        env_file,
        "FASTGEN_MODEL_DIRS",
        os.pathsep.join(str(path) for path in replacement),
    )
    return replacement


def discover_models(roots: Iterable[Path], *, model: str | None = None) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []
    for root in _dedupe_paths(roots):
        if not root.exists() or not root.is_dir():
            continue
        for path in _walk_dirs(root):
            markers = _markers(path)
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
            if model is None or candidate.model_family_guess in {model, "unknown"}:
                candidates.append(candidate)
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
    for current, dir_names, _file_names in os.walk(root):
        dir_names[:] = [
            name for name in dir_names if name not in SKIP_DIR_NAMES and not name.startswith(".")
        ]
        yield Path(current)


def _markers(path: Path) -> tuple[str, ...]:
    markers: list[str] = []
    try:
        children = list(path.iterdir())
    except OSError:
        return ()

    names = {child.name for child in children if child.is_file()}
    for marker in MODEL_MARKER_FILES:
        if marker in names:
            markers.append(marker)

    for child in children:
        if child.is_file() and child.suffix.lower() in MODEL_MARKER_SUFFIXES:
            markers.append(f"*{child.suffix.lower()}")
    return tuple(sorted(set(markers)))


def _candidate_id(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_env_value(env_file: Path, key: str, value: str) -> None:
    line = f"{key}={value}"
    if not env_file.exists():
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(line + "\n", encoding="utf-8")
        return

    lines = env_file.read_text(encoding="utf-8").splitlines()
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
