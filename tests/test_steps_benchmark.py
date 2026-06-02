from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


def test_steps_benchmark_defaults_do_not_start_real_mlx_work(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_base = tmp_path / "steps"
    env = {
        **os.environ,
        "FASTGEN_STEPS_OUTPUT_BASE": str(output_base),
    }
    env.pop("FASTGEN_STEPS_ALLOW_HEAVY", None)

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "steps_benchmark.py")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0
    assert "FASTGEN_STEPS_ALLOW_HEAVY=1" in result.stdout
    assert "MLX/Metal is unavailable" not in result.stdout

    records = [
        json.loads(line)
        for line in (output_base / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "steps": 1,
            "skipped": True,
            "error": (
                "skipped: real MLX benchmark requires FASTGEN_STEPS_ALLOW_HEAVY=1 "
                "after reviewing memory limits and model paths"
            ),
        }
    ]


def test_steps_benchmark_rejects_invalid_positive_integer_env(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_bad_env_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT_MAX_BYTES", "0")

    with pytest.raises(Exception, match="FASTGEN_STEPS_CHILD_RESULT_MAX_BYTES must be a positive integer"):
        spec.loader.exec_module(module)


def test_steps_benchmark_rejects_invalid_step_values(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_bad_steps_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,nope")

    with pytest.raises(Exception, match="FASTGEN_STEPS_VALUES must be a positive integer"):
        spec.loader.exec_module(module)


def test_steps_benchmark_heavy_mode_runs_step_in_child_process(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    child_result = module.OUTPUT_BASE / "steps_1.child.json"

    def fake_run(*args, **kwargs):
        assert kwargs.get("capture_output") is None
        assert "stdout" in kwargs
        kwargs["stdout"].write("child output\n")
        child_result.write_text(
            json.dumps({"steps": 1, "skipped": True, "error": "child guard"}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    with patch.object(module.subprocess, "run", side_effect=fake_run) as run:
        assert module.main() == 1

    run.assert_called_once()
    assert module.RESULTS_JSONL.exists()
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["steps"] == 1
    assert records[0]["skipped"] is True
    assert records[0]["error"] == "child guard"
    assert records[0]["log_path"].endswith("steps_1.child.log")


def test_steps_benchmark_child_timeout_records_abort(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_timeout_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)
    monkeypatch.setattr(module, "CHILD_TIMEOUT_SECONDS", 7)

    def fake_run(*_args, **_kwargs):
        _kwargs["stdout"].write("partial\n")
        raise subprocess.TimeoutExpired(cmd=["child"], timeout=7)

    with patch.object(module.subprocess, "run", side_effect=fake_run):
        assert module.main() == 1

    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "steps": 1,
            "error": "child process timed out after 7s",
            "aborted": True,
            "log_path": str(module.OUTPUT_BASE / "steps_1.child.log"),
        }
    ]


def test_steps_benchmark_child_abort_records_cleanup_status(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_abort_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(module.RuntimeMemoryAbort("runtime stop")),
    )
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {"mlx_cache_cleared": True})

    assert module.run_child() == 0

    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "steps": 1,
            "error": "runtime stop",
            "aborted": True,
            "cleanup": {"mlx_cache_cleared": True},
        }
    ]


def test_steps_benchmark_child_guard_block_marks_guard_blocked(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_guard_block_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(module.MemoryGuardError("low memory")),
    )

    assert module.run_child() == 0

    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "steps": 1,
            "error": "low memory",
            "skipped": True,
            "guard_blocked": True,
        }
    ]


def test_steps_benchmark_child_rejects_invalid_step_env(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_bad_step_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(tmp_path / "steps" / "child.json"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "0")
    spec.loader.exec_module(module)

    assert module.run_child() == 1


def test_steps_benchmark_child_rejects_result_path_outside_output_base(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_bad_path_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(tmp_path / "outside.json"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    assert module.run_child() == 1
    assert not (tmp_path / "outside.json").exists()


def test_steps_benchmark_run_single_does_not_configure_mlx_before_model_preflight(tmp_path, monkeypatch):
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_order_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    calls: list[str] = []
    monkeypatch.setattr(module, "check_memory_guard", lambda label: calls.append("system") or {"free_gb": 100})
    monkeypatch.setattr(
        module,
        "check_run_allocation_budget",
        lambda **kwargs: calls.append("budget") or {"shape_floor_gb": 1},
    )
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "check_text_prompt_budget", lambda **kwargs: calls.append("prompt"))
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {"freed_gb": 0})
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)

    class FakePipeline:
        def load_model(self):
            calls.append("model_preflight")
            raise module.MemoryGuardError("model preflight blocked")

    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())

    with pytest.raises(module.MemoryGuardError, match="model preflight blocked"):
        module.run_single(1)

    assert calls == ["system", "budget", "prompt", "model_preflight"]


def test_steps_benchmark_run_single_checks_dependency_before_mlx_import(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_dependency_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None if name == "mlx_video" else object())

    with pytest.raises(module.MemoryGuardError, match="dependency unavailable before MLX import"):
        module.run_single(1)


def test_steps_benchmark_run_single_model_preflight_can_block_before_mlx_import(tmp_path, monkeypatch):
    import builtins
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_model_preflight_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    class FakePipeline:
        def load_model(self):
            raise module.MemoryGuardError("model asset preflight blocked")

    def guarded_import(name, *args, **kwargs):
        if name in {"mlx", "mlx.core"}:
            raise AssertionError("mlx.core must not be imported before model preflight")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    sys.modules.pop("mlx", None)
    sys.modules.pop("mlx.core", None)
    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(module.MemoryGuardError, match="model asset preflight blocked"):
        module.run_single(1)


def test_steps_benchmark_run_single_preflights_png_frame_save(tmp_path, monkeypatch):
    import importlib.util
    import numpy as np
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_png_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class FakePipeline:
        def load_model(self):
            return {}

        def prepare_prompt(self, *, prompt, negative_prompt):
            return {"prompt": prompt, "negative_prompt": negative_prompt}

        def encode_text(self, prepared):
            return object()

        def init_latents(self, *, seed, width, height, frames):
            return object()

        def denoise_step(self, latents, *, step_index, steps, guidance, cache):
            return latents

        def decode(self, latents):
            return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    fake_mx = types.SimpleNamespace(
        eval=lambda *args: None,
        array=lambda value: value,
    )
    saved: list[str] = []

    class FakeImage:
        def save(self, path):
            saved.append(path)

    fake_image_module = types.SimpleNamespace(fromarray=lambda frame: FakeImage())
    host_checks: list[tuple[int, str]] = []

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {"freed_gb": 0, "free_after_gb": 100})
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(
        module,
        "check_host_allocation_headroom",
        lambda required, *, label: host_checks.append((required, label)),
    )
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    result = module.run_single(1)

    assert result["video_shape"] == [2, 4, 4, 3]
    assert host_checks == [(2 * 4 * 4 * 3 * 2, "steps_1 png frames")]
    assert len(saved) == 2


def test_steps_benchmark_rejects_unexpected_video_shape_before_png_save(tmp_path, monkeypatch):
    import importlib.util
    import numpy as np
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_video_shape_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class FakePipeline:
        def load_model(self):
            return {}

        def prepare_prompt(self, *, prompt, negative_prompt):
            return {"prompt": prompt, "negative_prompt": negative_prompt}

        def encode_text(self, prepared):
            return object()

        def init_latents(self, *, seed, width, height, frames):
            return object()

        def denoise_step(self, latents, *, step_index, steps, guidance, cache):
            return latents

        def decode(self, latents):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    fake_image_module = types.SimpleNamespace(
        fromarray=lambda frame: (_ for _ in ()).throw(AssertionError("PNG save must be rejected first"))
    )

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    with pytest.raises(RuntimeError, match="decoded benchmark video must have shape"):
        module.run_single(1)


def test_steps_benchmark_heavy_guard_error_returns_nonzero(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_exit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)
    monkeypatch.setattr(
        module,
        "run_step_in_child",
        lambda steps: {"steps": steps, "skipped": True, "error": "guard blocked"},
    )

    assert module.main() == 1


def test_steps_benchmark_does_not_read_stale_child_result(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_stale_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)
    module.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    stale = module.OUTPUT_BASE / "steps_1.child.json"
    stale.write_text(json.dumps({"steps": 1, "error": "stale"}) + "\n", encoding="utf-8")

    def fake_run(*args, **_kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7)

    with patch.object(module.subprocess, "run", side_effect=fake_run):
        result = module.run_step_in_child(1)

    assert result == {
        "steps": 1,
        "error": "child process exited 7 without a result record",
        "aborted": True,
        "log_path": str(module.OUTPUT_BASE / "steps_1.child.log"),
    }


def test_steps_benchmark_rejects_malformed_child_result_as_abort(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_bad_child_result_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)
    module.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    child_result = module.OUTPUT_BASE / "steps_1.child.json"

    def fake_run(*args, **_kwargs):
        child_result.write_text("{not-json}\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    with patch.object(module.subprocess, "run", side_effect=fake_run):
        result = module.run_step_in_child(1)

    assert result["steps"] == 1
    assert result["aborted"] is True
    assert result["error"].startswith("child result file is not valid JSONL:")
    assert result["log_path"] == str(module.OUTPUT_BASE / "steps_1.child.log")


def test_steps_benchmark_rejects_oversized_child_result(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_big_result_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)
    module.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "CHILD_RESULT_MAX_BYTES", 10)

    child_result = module.OUTPUT_BASE / "steps_1.child.json"

    def fake_run(*args, **_kwargs):
        child_result.write_text("x" * 11, encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    with patch.object(module.subprocess, "run", side_effect=fake_run):
        result = module.run_step_in_child(1)

    assert result == {
        "steps": 1,
        "error": "child result file is 11 bytes, exceeds limit 10 bytes",
        "aborted": True,
        "log_path": str(module.OUTPUT_BASE / "steps_1.child.log"),
    }


def test_steps_benchmark_heavy_mode_recovers_between_child_processes(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_recovery_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    launched: list[int] = []
    recovered: list[str] = []

    def fake_child(steps):
        launched.append(steps)
        return {
            "steps": steps,
            "denoise_total_s": 1.0,
            "vae_decode_s": 1.0,
            "pixel_min": 0,
            "pixel_max": 255,
            "pixel_mean": 127.0,
        }

    def fake_recovery(label):
        recovered.append(label)
        return {"free_gb": 100}

    monkeypatch.setattr(module, "run_step_in_child", fake_child)
    monkeypatch.setattr(module, "parent_inter_child_recovery", fake_recovery)

    assert module.main() == 0

    assert launched == [1, 2]
    assert recovered == ["pre-steps_2"]


def test_steps_benchmark_heavy_mode_does_not_count_skipped_child_as_completed_run(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_skip_counter_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    launched: list[int] = []
    recovered: list[str] = []

    def fake_child(steps):
        launched.append(steps)
        return {"steps": steps, "skipped": True, "error": f"child {steps}"}

    monkeypatch.setattr(module, "run_step_in_child", fake_child)
    monkeypatch.setattr(module, "parent_inter_child_recovery", lambda label: recovered.append(label) or {"free_gb": 100})

    assert module.main() == 1

    assert launched == [1, 2]
    assert recovered == []


def test_steps_benchmark_multiple_heavy_runs_require_extra_opt_in(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_multi_heavy_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    monkeypatch.delenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", raising=False)
    spec.loader.exec_module(module)

    launched: list[int] = []
    monkeypatch.setattr(
        module,
        "run_step_in_child",
        lambda steps: launched.append(steps) or {"steps": steps},
    )

    assert module.main() == 1

    assert launched == []
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["steps"] for record in records] == [1, 2]
    assert all(record["skipped"] is True for record in records)
    assert all("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY=1" in record["error"] for record in records)


def test_steps_benchmark_heavy_mode_stops_after_child_abort(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_stop_after_abort_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    launched: list[int] = []

    def fake_child(steps):
        launched.append(steps)
        return {"steps": steps, "aborted": True, "error": "runtime memory abort"}

    monkeypatch.setattr(module, "run_step_in_child", fake_child)
    monkeypatch.setattr(module, "parent_inter_child_recovery", lambda label: {"free_gb": 100})

    assert module.main() == 1

    assert launched == [1]
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"steps": 1, "aborted": True, "error": "runtime memory abort"}]


def test_steps_benchmark_heavy_mode_stops_after_child_guard_block(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_stop_after_guard_block_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    launched: list[int] = []

    def fake_child(steps):
        launched.append(steps)
        return {"steps": steps, "skipped": True, "guard_blocked": True, "error": "low memory"}

    monkeypatch.setattr(module, "run_step_in_child", fake_child)
    monkeypatch.setattr(module, "parent_inter_child_recovery", lambda label: {"free_gb": 100})

    assert module.main() == 1

    assert launched == [1]
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"steps": 1, "skipped": True, "guard_blocked": True, "error": "low memory"}]
