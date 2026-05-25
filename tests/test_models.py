from __future__ import annotations

import os

from fastgen_profiler.models import (
    discover_import_dirs,
    discover_models,
    merge_model_dirs_into_env,
    model_dirs_from_sources,
    replace_model_dirs_in_env,
)


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

    assert dirs == [env_dir, family_dir, cli_dir]


def test_discovers_default_import_dirs_for_supported_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = [
        tmp_path / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
        tmp_path / "Documents/Draw Things/Models",
        tmp_path / "ComfyUI/models",
        tmp_path / "Documents/ComfyUI/models",
        tmp_path / ".cache/huggingface/hub",
        tmp_path / ".cache/lm-studio/models",
        tmp_path / "Library/Application Support/LM Studio/models",
        tmp_path / ".ollama/models",
    ]
    for path in expected:
        path.mkdir(parents=True)

    found = discover_import_dirs("all")

    assert found == [path.resolve() for path in expected]


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
