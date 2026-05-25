from __future__ import annotations

import json
import sys

from fastgen_profiler.cli import main


def test_cli_run_parses_required_arguments_and_stub_writes_jsonl(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "a test prompt",
            "--negative-prompt",
            "blur",
            "--seed",
            "7",
            "--width",
            "512",
            "--height",
            "288",
            "--frames",
            "16",
            "--fps",
            "12",
            "--steps",
            "4",
            "--guidance",
            "3.5",
            "--quant",
            "q8",
            "--cache",
            "prompt",
            "--compile",
            "off",
            "--output-dir",
            str(output_dir),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert records
    assert {record["phase"] for record in records} >= {"model_load", "denoise_step", "total"}
    assert records[0]["model"] == "wan2.2"
    assert records[0]["backend"] == "stub"
    assert records[0]["negative_prompt_hash"]
    assert all(record["output_path"] is None for record in records)


def test_cli_save_video_writes_placeholder_and_records_output_path(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "run",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "video",
            "--negative-prompt",
            "",
            "--seed",
            "1",
            "--width",
            "320",
            "--height",
            "180",
            "--frames",
            "8",
            "--fps",
            "8",
            "--steps",
            "2",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "on",
            "--output-dir",
            str(output_dir),
            "--result-jsonl",
            str(jsonl_path),
            "--save-video",
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    output_paths = [record["output_path"] for record in records if record["output_path"]]
    assert output_paths
    assert output_paths[0].endswith(".stub.mp4")


def test_cli_run_appends_jsonl_records(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    args = [
        "run",
        "--model",
        "wan2.2",
        "--backend",
        "stub",
        "--prompt",
        "append",
        "--negative-prompt",
        "",
        "--seed",
        "2",
        "--width",
        "128",
        "--height",
        "128",
        "--frames",
        "2",
        "--fps",
        "2",
        "--steps",
        "1",
        "--guidance",
        "1.0",
        "--quant",
        "none",
        "--cache",
        "none",
        "--compile",
        "off",
        "--output-dir",
        str(tmp_path / "outputs"),
        "--result-jsonl",
        str(jsonl_path),
        "--no-save-video",
        "--dry-run",
    ]

    assert main(args) == 0
    first_count = len(_read_jsonl(jsonl_path))
    assert main(args) == 0

    assert len(_read_jsonl(jsonl_path)) == first_count * 2


def test_report_command_produces_markdown_from_jsonl(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    report_path = tmp_path / "report.md"

    main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "report",
            "--negative-prompt",
            "",
            "--seed",
            "2",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "2",
            "--guidance",
            "2.0",
            "--quant",
            "q4",
            "--cache",
            "all",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    exit_code = main(["report", "--input", str(jsonl_path), "--output", str(report_path)])

    assert exit_code == 0
    report = report_path.read_text(encoding="utf-8")
    assert "Total Time By Run" in report
    assert "average denoise step time" in report
    assert "Recommended Next Bottleneck" in report


def test_profile_command_runs_full_wan_suite_and_writes_comparison_report(tmp_path, capsys):
    results_dir = tmp_path / "profiles"

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "profile suite",
            "--seed",
            "11",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--results-dir",
            str(results_dir),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    jsonl_files = list(results_dir.glob("*T*_wan2.2.jsonl"))
    assert exit_code == 0
    assert len(jsonl_files) == 1
    report_path = jsonl_files[0].with_suffix(".md")
    assert report_path.exists()
    records = _read_jsonl(jsonl_files[0])
    totals = [record for record in records if record["phase"] == "total"]
    assert {record["preset"] for record in totals} == {
        "smoke",
        "small-baseline",
        "quality-threshold",
        "cache-experiment",
        "compile-experiment",
        "stress",
    }
    assert all(record["profile_id"] for record in records)
    assert all(record["profile_name"] == "wan2.2-full-preset-suite" for record in records)
    assert "Profile suite summary" in output
    assert "Recommended next bottleneck" in output
    report = report_path.read_text(encoding="utf-8")
    assert "Preset Comparison" in report
    assert "average denoise step" in report


def test_profile_command_skips_ltx23_stress(tmp_path, capsys):
    jsonl_path = tmp_path / "ltx-profile.jsonl"

    exit_code = main(
        [
            "profile",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "profile suite",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    records = _read_jsonl(jsonl_path)
    stress_records = [record for record in records if record["preset"] == "stress"]
    assert exit_code == 0
    assert stress_records
    assert all(record["error"].startswith("skipped:") for record in stress_records)
    assert "skipped" in output


def test_mlx_scaffold_writes_failed_schema_records(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx scaffold",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert records
    assert any(record["error"] for record in records)
    assert all(record["backend"] == "mlx" for record in records)
    assert all(record["model_path"] == str(model_path.resolve()) for record in records)


def test_smoke_preset_applies_requested_shape_and_defaults(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "preset smoke",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    total = [record for record in records if record["phase"] == "total"]
    assert len(total) == 1
    assert total[0]["width"] == 384
    assert total[0]["height"] == 384
    assert total[0]["frames"] == 16
    assert total[0]["steps"] == 8
    assert total[0]["cache"] == "none"
    assert total[0]["compile"] == "off"
    assert all(record["output_path"] is None for record in records)


def test_small_baseline_preset_appends_guidance_and_quant_variants(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "small-baseline",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "small baseline",
            "--seed",
            "2",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert len(totals) == 4
    assert {(record["guidance"], record["quant"]) for record in totals} == {
        (1.0, "none"),
        (1.0, "q8p"),
        (3.5, "none"),
        (3.5, "q8p"),
    }
    assert all(record["width"] == 512 and record["height"] == 512 for record in totals)


def test_quality_threshold_preset_saves_video_for_step_variants(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "quality-threshold",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "quality threshold",
            "--seed",
            "3",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert [record["steps"] for record in totals] == [16, 24, 32, 40]
    assert all(record["output_path"] for record in totals)


def test_cache_and_compile_presets_expand_variants(tmp_path):
    cache_jsonl = tmp_path / "cache.jsonl"
    compile_jsonl = tmp_path / "compile.jsonl"
    common_args = [
        "--model",
        "wan2.2",
        "--backend",
        "stub",
        "--prompt",
        "variant",
        "--seed",
        "4",
        "--output-dir",
        str(tmp_path / "outputs"),
    ]

    assert main(["run", "--preset", "cache-experiment", *common_args, "--result-jsonl", str(cache_jsonl)]) == 0
    assert main(["run", "--preset", "compile-experiment", *common_args, "--result-jsonl", str(compile_jsonl)]) == 0

    cache_totals = [record for record in _read_jsonl(cache_jsonl) if record["phase"] == "total"]
    compile_totals = [record for record in _read_jsonl(compile_jsonl) if record["phase"] == "total"]
    assert [record["cache"] for record in cache_totals] == ["none", "prompt", "feature", "all"]
    assert [record["compile"] for record in compile_totals] == ["off", "on"]


def test_stress_preset_rejects_ltx23_until_backend_stabilizes(tmp_path):
    try:
        main(
            [
                "run",
                "--preset",
                "stress",
                "--model",
                "ltx2.3",
                "--backend",
                "stub",
                "--prompt",
                "stress",
                "--seed",
                "5",
                "--output-dir",
                str(tmp_path / "outputs"),
                "--result-jsonl",
                str(tmp_path / "stress.jsonl"),
            ]
        )
    except SystemExit as exc:
        assert "wan2.2" in str(exc)
    else:
        raise AssertionError("stress preset should reject ltx2.3")


def test_missing_preset_prompts_for_selection_when_interactive(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "1")

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "interactive preset",
            "--seed",
            "6",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert len(totals) == 1
    assert totals[0]["width"] == 384


def test_models_list_uses_env_and_cli_model_dirs(tmp_path, capsys):
    env_root = tmp_path / "env-root"
    cli_root = tmp_path / "cli-root"
    env_model = env_root / "wan-env"
    cli_model = cli_root / "wan-cli"
    ltx_model = env_root / "ltx-env"
    env_model.mkdir(parents=True)
    cli_model.mkdir(parents=True)
    ltx_model.mkdir(parents=True)
    (env_model / "config.json").write_text("{}", encoding="utf-8")
    (cli_model / "model.safetensors").write_text("", encoding="utf-8")
    (ltx_model / "model.safetensors").write_text("", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(f"FASTGEN_MODEL_DIRS={env_root}\n", encoding="utf-8")

    exit_code = main(
        [
            "models",
            "list",
            "--env-file",
            str(env_file),
            "--model-dir",
            str(cli_root),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-env" in output
    assert "wan-cli" in output
    assert "ltx-env" in output


def test_run_records_direct_model_path(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "direct-wan"
    model_path.mkdir()

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--model-path",
            str(model_path),
            "--prompt",
            "direct model",
            "--seed",
            "7",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert records
    assert all(record["model_path"] == str(model_path.resolve()) for record in records)
    assert all(record["model_id"] == model_path.name for record in records)


def test_run_selects_model_id_from_env_dirs(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_root = tmp_path / "models"
    model_path = model_root / "nested" / "wan-local"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--model-id",
            "nested/wan-local",
            "--env-file",
            str(env_file),
            "--prompt",
            "model id",
            "--seed",
            "8",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert all(record["model_id"] == "nested/wan-local" for record in records)
    assert all(record["model_source_root"] == str(model_root.resolve()) for record in records)


def test_interactive_mlx_model_selection(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_root = tmp_path / "models"
    first = model_root / "wan-first"
    second = model_root / "wan-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.json").write_text("{}", encoding="utf-8")
    (second / "config.json").write_text("{}", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "2")

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-dir",
            str(model_root),
            "--prompt",
            "interactive model",
            "--seed",
            "9",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert all(record["model_id"] == "wan-second" for record in records)
    assert any("wan-second" in record["error"] for record in records if record["error"])


def test_non_interactive_mlx_without_model_selection_writes_error_record(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--prompt",
            "missing model",
            "--seed",
            "10",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "model_load"
    assert records[0]["model_path"] is None
    assert "model selection required" in records[0]["error"]


def test_models_import_dry_run_discovers_dirs_without_writing_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    hf_dir = tmp_path / ".cache/huggingface/hub"
    hf_model = hf_dir / "models--owner--wan2.2" / "snapshots" / "abc123"
    hf_model.mkdir(parents=True)
    (hf_model / "model_index.json").write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "all",
            "--env-file",
            str(env_file),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Discovered app roots" in output
    assert "Generation model directories to register" in output
    assert str(hf_dir.resolve()) in output
    assert str(hf_model.resolve()) in output
    assert "Dry run" in output
    assert not env_file.exists()


def test_models_import_writes_env_non_interactive(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    comfy_dir = tmp_path / "ComfyUI/models"
    comfy_model = comfy_dir / "diffusion_models" / "wan2.2-video"
    old_dir = tmp_path / "old-models"
    comfy_model.mkdir(parents=True)
    (comfy_model / "model.safetensors").write_text("", encoding="utf-8")
    old_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"# existing\nFASTGEN_MODEL_DIRS={old_dir}\nOTHER=value\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "comfyui",
            "--env-file",
            str(env_file),
        ]
    )

    content = env_file.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# existing" in content
    assert "OTHER=value" in content
    assert f"FASTGEN_MODEL_DIRS={comfy_model.resolve()}" in content
    assert str(old_dir) not in content


def test_models_import_fails_when_roots_have_no_generation_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    lmstudio_dir = tmp_path / ".cache/lm-studio/models"
    llm_dir = lmstudio_dir / "owner" / "chat-model"
    llm_dir.mkdir(parents=True)
    (llm_dir / "model.gguf").write_text("", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=value\n", encoding="utf-8")

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "lmstudio",
            "--env-file",
            str(env_file),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "No Wan2.2/LTX2.3 generation model candidates found" in output
    assert env_file.read_text(encoding="utf-8") == "OTHER=value\n"


def test_models_import_fails_when_no_directories_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    env_file = tmp_path / ".env"

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "all",
            "--env-file",
            str(env_file),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "No known model directories found." in output
    assert not env_file.exists()


def test_interactive_main_menu_can_import_model_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    draw_dir = tmp_path / "Documents/Draw Things/Models"
    draw_model = draw_dir / "wan2.2-draw"
    draw_model.mkdir(parents=True)
    (draw_model / "model.safetensors").write_text("", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["3", "y"])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    assert exit_code == 0
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert str(draw_model.resolve()) in content


def test_interactive_main_menu_run_profile_creates_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["1", "", "", "", "", "", "", ""])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    assert exit_code == 0
    records = _read_jsonl(tmp_path / "artifacts/results.jsonl")
    assert records
    assert records[0]["backend"] == "stub"
    assert records[0]["model"] == "wan2.2"


def test_interactive_main_menu_list_models_outputs_all_candidates_without_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model_root = tmp_path / "models"
    wan_path = model_root / "wan-menu"
    ltx_path = model_root / "ltx-menu"
    wan_path.mkdir(parents=True)
    ltx_path.mkdir(parents=True)
    (wan_path / "config.json").write_text("{}", encoding="utf-8")
    (ltx_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["2"])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-menu" in output
    assert "ltx-menu" in output


def test_run_command_prompts_for_missing_required_values_interactively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["", "", "", "", "", "", ""])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main(["run"])

    assert exit_code == 0
    assert (tmp_path / "artifacts/results.jsonl").exists()


def test_models_command_without_subcommand_lists_interactively(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model_root = tmp_path / "models"
    model_path = model_root / "wan-models-command"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "")

    exit_code = main(["models"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-models-command" in output


def test_models_list_without_model_outputs_all_candidates(tmp_path, capsys):
    model_root = tmp_path / "models"
    wan_path = model_root / "wan-list-command"
    ltx_path = model_root / "ltx-list-command"
    wan_path.mkdir(parents=True)
    ltx_path.mkdir(parents=True)
    (wan_path / "config.json").write_text("{}", encoding="utf-8")
    (ltx_path / "config.json").write_text("{}", encoding="utf-8")

    exit_code = main(["models", "list", "--model-dir", str(model_root)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-list-command" in output
    assert "ltx-list-command" in output


def test_run_command_missing_required_values_fails_non_interactively():
    try:
        main(["run"])
    except SystemExit as exc:
        assert "--model" in str(exc)
    else:
        raise AssertionError("run without required values should fail when non-interactive")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
