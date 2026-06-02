from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

import fastgen_profiler.models as models_module
from fastgen_profiler.models import (
    discover_generation_model_dirs,
    discover_import_dirs,
    discover_models,
    load_env_file,
    merge_model_dirs_into_env,
    model_dirs_from_sources,
    replace_model_dirs_in_env,
)


def test_model_discovery_does_not_materialize_entire_directory_list():
    source = Path("src/fastgen_profiler/models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "list" or not node.args:
            continue
        arg = node.args[0]
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "iterdir"
        ):
            offenders.append(node.lineno)

    assert offenders == []


def test_model_discovery_rejects_excessive_directory_traversal(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    walked = [root / "wan-one", root / "wan-two"]

    monkeypatch.setattr(models_module, "_walk_dirs", lambda path: iter(walked))
    monkeypatch.setattr(models_module, "_markers", lambda path, **kwargs: ())

    with pytest.raises(ValueError, match="visited more than 1 directories"):
        discover_models([root], model="wan2.2", max_dirs=1)


def test_model_discovery_rejects_excessive_candidate_accumulation(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    walked = [root / "wan-one", root / "wan-two"]

    monkeypatch.setattr(models_module, "_walk_dirs", lambda path: iter(walked))
    monkeypatch.setattr(models_module, "_markers", lambda path, **kwargs: ("config.json",))

    with pytest.raises(ValueError, match="found more than 1 candidates"):
        discover_models([root], model="wan2.2", max_candidates=1)


def test_model_discovery_rejects_excessive_marker_file_scan(tmp_path):
    root = tmp_path / "models"
    model_dir = root / "wan2.2-large-dir"
    model_dir.mkdir(parents=True)
    for index in range(3):
        (model_dir / f"nonmatch-{index}.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="scanned more than 2 files"):
        discover_models([root], model="wan2.2", max_files=2)


def test_model_discovery_rejects_excessive_flat_file_scan(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    for index in range(3):
        (root / f"nonmatch-{index}.txt").write_text("", encoding="utf-8")

    monkeypatch.setattr(models_module, "_walk_dirs", lambda path: iter(()))

    with pytest.raises(ValueError, match="scanned more than 2 files"):
        discover_models([root], model="wan2.2", max_files=2)


def test_model_marker_scan_ignores_iteration_time_filesystem_interrupt():
    class FakeChild:
        name = "config.json"
        suffix = ".json"

        def is_file(self):
            return True

    class FakePath:
        def __str__(self):
            return "fake-model-dir"

        def iterdir(self):
            yield FakeChild()
            raise InterruptedError("scan interrupted")

    assert models_module._markers(FakePath()) == ("config.json",)


def test_discovers_hugging_face_snapshot_and_draw_things_style_models(tmp_path):
    hf_model = tmp_path / "huggingface" / "models--owner--wan2.2" / "snapshots" / "abc123"
    draw_model = tmp_path / "Draw Things" / "Models" / "ltx2.3-video"
    hf_model.mkdir(parents=True)
    draw_model.mkdir(parents=True)
    (hf_model / "model_index.json").write_text("{}", encoding="utf-8")
    (draw_model / "model.safetensors").write_text("", encoding="utf-8")

    wan_candidates = discover_models([tmp_path], model="wan2.2")
    ltx_candidates = discover_models([tmp_path], model="ltx2.3")

    assert any(candidate.model_family_guess == "wan2.2" for candidate in wan_candidates)
    assert any(candidate.model_family_guess == "ltx2.3" for candidate in ltx_candidates)
    assert any("model_index.json" in candidate.markers for candidate in wan_candidates)
    assert any("*.safetensors" in candidate.markers for candidate in ltx_candidates)


def test_model_specific_discovery_excludes_unknown_and_gguf_only_candidates(tmp_path):
    wan_model = tmp_path / "wan2.2-video"
    unknown_model = tmp_path / "random-video-model"
    gguf_model = tmp_path / "wan2.2-llm"
    subcomponent = tmp_path / "models--owner--wan2.2" / "snapshots" / "abc" / "audio_vae"
    wan_model.mkdir()
    unknown_model.mkdir()
    gguf_model.mkdir()
    subcomponent.mkdir(parents=True)
    (wan_model / "config.json").write_text("{}", encoding="utf-8")
    (unknown_model / "config.json").write_text("{}", encoding="utf-8")
    (gguf_model / "model.gguf").write_text("", encoding="utf-8")
    (subcomponent / "config.json").write_text("{}", encoding="utf-8")

    wan_candidates = discover_models([tmp_path], model="wan2.2")

    assert [candidate.name for candidate in wan_candidates] == ["wan2.2-video"]


def test_generation_model_dirs_returns_leaf_wan_ltx_directories(tmp_path):
    root = tmp_path / "ComfyUI/models"
    wan_model = root / "diffusion_models" / "wan2.2-video"
    ltx_model = root / "checkpoints" / "ltx2.3-video"
    llm_model = root / "llm" / "wan2.2-chat"
    wan_model.mkdir(parents=True)
    ltx_model.mkdir(parents=True)
    llm_model.mkdir(parents=True)
    (wan_model / "model.safetensors").write_text("", encoding="utf-8")
    (ltx_model / "model.ckpt").write_text("", encoding="utf-8")
    (llm_model / "model.gguf").write_text("", encoding="utf-8")

    dirs = discover_generation_model_dirs([root])

    assert dirs == [wan_model.resolve(), ltx_model.resolve()]


def test_model_dirs_from_env_and_cli_dirs_are_combined(tmp_path):
    env_dir = tmp_path / "env-models"
    family_dir = tmp_path / "wan-models"
    cli_dir = tmp_path / "cli-models"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"FASTGEN_MODEL_DIRS={env_dir}\nFASTGEN_MODEL_DIR_WAN22={family_dir}\n",
        encoding="utf-8",
    )

    dirs = model_dirs_from_sources(
        model="wan2.2",
        cli_dirs=[cli_dir],
        env_file=env_file,
    )

    # env dirs and cli dir must all be present (auto-discovered dirs may also appear)
    assert env_dir in dirs
    assert family_dir in dirs
    assert cli_dir in dirs


def test_load_env_file_rejects_oversized_file_before_parsing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FASTGEN_MODEL_DIRS=/models\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds env file read limit"):
        load_env_file(env_file, max_bytes=8)


def test_merge_model_dirs_rejects_oversized_existing_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=value\n", encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="exceeds env file read limit"):
        merge_model_dirs_into_env(env_file, [model_dir], max_bytes=8)


def test_discovers_default_import_dirs_for_supported_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = [
        tmp_path / "Library/Containers/Draw Things/Data/Documents/Models",
        tmp_path / "Library/Containers/Draw Things/Data",
        tmp_path / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
        tmp_path / "Library/Containers/com.liuliu.draw-things/Data",
        tmp_path / "Documents/Draw Things/Models",
        tmp_path / "ComfyUI/models",
        tmp_path / "Documents/ComfyUI/models",
        tmp_path / ".cache/huggingface/hub",
        tmp_path / ".cache/lm-studio/models",
        tmp_path / "Library/Application Support/LM Studio/models",
        tmp_path / ".ollama/models",
    ]
    for path in expected:
        path.mkdir(parents=True, exist_ok=True)

    found = discover_import_dirs("all")

    assert found == [path.resolve() for path in expected]


def test_draw_things_container_data_root_can_import_nested_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data_root = tmp_path / "Library/Containers/Draw Things/Data"
    model_dir = data_root / "Documents/Models/wan2.2-drawthings"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")

    roots = discover_import_dirs("drawthings")
    generation_dirs = discover_generation_model_dirs(roots)

    assert data_root.resolve() in roots
    assert model_dir.resolve() in generation_dirs


def test_merge_model_dirs_into_env_preserves_existing_content_and_dedupes(tmp_path):
    existing = tmp_path / "existing"
    new_dir = tmp_path / "new"
    existing.mkdir()
    new_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"# keep me\nOTHER=value\nFASTGEN_MODEL_DIRS={existing}\n",
        encoding="utf-8",
    )

    merged = merge_model_dirs_into_env(env_file, [existing, new_dir])

    content = env_file.read_text(encoding="utf-8")
    assert merged == [existing, new_dir.resolve()]
    assert "# keep me" in content
    assert "OTHER=value" in content
    assert f"FASTGEN_MODEL_DIRS={existing}{os.pathsep}{new_dir.resolve()}" in content


def test_merge_model_dirs_into_env_creates_missing_env_file(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    env_file = tmp_path / ".env"

    merged = merge_model_dirs_into_env(env_file, [model_dir])

    assert merged == [model_dir.resolve()]
    assert env_file.read_text(encoding="utf-8") == f"FASTGEN_MODEL_DIRS={model_dir.resolve()}\n"


def test_replace_model_dirs_in_env_clears_existing_fastgen_model_dirs(tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"# keep me\nFASTGEN_MODEL_DIRS={old_dir}\nOTHER=value\n",
        encoding="utf-8",
    )

    replacement = replace_model_dirs_in_env(env_file, [new_dir, new_dir])

    content = env_file.read_text(encoding="utf-8")
    assert replacement == [new_dir.resolve()]
    assert "# keep me" in content
    assert "OTHER=value" in content
    assert str(old_dir) not in content
    assert f"FASTGEN_MODEL_DIRS={new_dir.resolve()}" in content
