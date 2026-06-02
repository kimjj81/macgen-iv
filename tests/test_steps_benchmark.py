from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
import weakref
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


def test_steps_benchmark_rejects_oversized_numeric_env_before_int(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_huge_numeric_env_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "1" * 10_000)

    with pytest.raises(Exception, match="FASTGEN_STEPS_WIDTH must be no longer than 64 chars"):
        spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("env_name", "env_value", "message"),
    [
        (
            "FASTGEN_STEPS_WIDTH",
            "4097",
            "FASTGEN_STEPS_WIDTH must be no greater than 4096",
        ),
        (
            "FASTGEN_STEPS_HEIGHT",
            "4097",
            "FASTGEN_STEPS_HEIGHT must be no greater than 4096",
        ),
        (
            "FASTGEN_STEPS_FRAMES",
            "258",
            "FASTGEN_STEPS_FRAMES must be no greater than 257",
        ),
        (
            "FASTGEN_STEPS_FPS",
            "241",
            "FASTGEN_STEPS_FPS must be no greater than 240",
        ),
        (
            "FASTGEN_STEPS_SEED",
            str(2**32),
            "FASTGEN_STEPS_SEED must be no greater than 4294967295",
        ),
        (
            "FASTGEN_MAX_PROMPT_CHARS",
            "65537",
            "FASTGEN_MAX_PROMPT_CHARS must be no greater than 65536",
        ),
        (
            "FASTGEN_STEPS_CHILD_TIMEOUT_SECONDS",
            str(24 * 60 * 60 + 1),
            "FASTGEN_STEPS_CHILD_TIMEOUT_SECONDS must be no greater than 86400",
        ),
        (
            "FASTGEN_STEPS_VALUES",
            "513",
            "FASTGEN_STEPS_VALUES must be no greater than 512",
        ),
        (
            "FASTGEN_STEPS_CHILD_RESULT_MAX_BYTES",
            str(1024 * 1024 + 1),
            "FASTGEN_STEPS_CHILD_RESULT_MAX_BYTES must be no greater than 1048576",
        ),
        (
            "FASTGEN_STEPS_RESULT_RECORD_MAX_BYTES",
            str(1024 * 1024 + 1),
            "FASTGEN_STEPS_RESULT_RECORD_MAX_BYTES must be no greater than 1048576",
        ),
        (
            "FASTGEN_STEPS_CHILD_LOG_TAIL_BYTES",
            str(1024 * 1024 + 1),
            "FASTGEN_STEPS_CHILD_LOG_TAIL_BYTES must be no greater than 1048576",
        ),
    ],
)
def test_steps_benchmark_rejects_unbounded_child_io_env(tmp_path, monkeypatch, env_name, env_value, message):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_unbounded_child_io_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(Exception, match=message):
        spec.loader.exec_module(module)


def test_steps_benchmark_rejects_too_many_step_values(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_many_steps_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", ",".join(str(index + 1) for index in range(17)))

    with pytest.raises(Exception, match="FASTGEN_STEPS_VALUES may contain at most 16 values"):
        spec.loader.exec_module(module)


def test_steps_benchmark_rejects_oversized_step_values_before_split(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_huge_steps_env_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1," * 10_000)

    with pytest.raises(Exception, match="FASTGEN_STEPS_VALUES must be no longer than 1039 chars"):
        spec.loader.exec_module(module)


@pytest.mark.parametrize("env_name", ["FASTGEN_STEPS_PROMPT", "FASTGEN_STEPS_NEGATIVE_PROMPT"])
def test_steps_benchmark_rejects_oversized_prompt_env_at_import(tmp_path, monkeypatch, env_name):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_large_prompt_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "8")
    if env_name == "FASTGEN_STEPS_NEGATIVE_PROMPT":
        monkeypatch.setenv("FASTGEN_STEPS_PROMPT", "short")
    monkeypatch.setenv(env_name, "x" * 9)

    with pytest.raises(Exception, match=f"{env_name} must be no longer than 8 chars"):
        spec.loader.exec_module(module)


def test_steps_benchmark_rejects_unbounded_prompt_limit_at_import(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_prompt_limit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "1000000")

    with pytest.raises(Exception, match="FASTGEN_MAX_PROMPT_CHARS must be no greater than"):
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
    child_log = module.OUTPUT_BASE / "steps_1.child.log"
    child_result = module.OUTPUT_BASE / "steps_1.child.json"
    assert records == [
        {
            "steps": 1,
            "error": "child process timed out after 7s",
            "aborted": True,
            "log_path": str(child_log),
        }
    ]
    assert not child_log.exists()
    assert not child_result.exists()


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


def test_steps_benchmark_child_abort_records_cleanup_failure(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_cleanup_failure_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    cleanup_errors: list[RuntimeError] = []

    def cleanup():
        exc = RuntimeError("cleanup failed")
        cleanup_errors.append(exc)
        raise exc

    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(module.RuntimeMemoryAbort("runtime stop")),
    )
    monkeypatch.setattr(module, "mlx_cleanup", cleanup)

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
            "cleanup": {
                "mlx_loaded": None,
                "mlx_cache_cleared": False,
                "mlx_cleanup_error": "mlx_cleanup raised: cleanup failed",
            },
        }
    ]
    assert cleanup_errors
    assert cleanup_errors[0].__traceback__ is None
    assert cleanup_errors[0].__cause__ is None
    assert cleanup_errors[0].__context__ is None


def test_steps_benchmark_child_abort_with_completed_cleanup_does_not_cleanup_again(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_abort_done_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    abort = module.RuntimeMemoryAbort("runtime stop")
    setattr(abort, module._CLEANUP_DONE_ATTR, True)
    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(abort),
    )
    monkeypatch.setattr(
        module,
        "mlx_cleanup",
        lambda: (_ for _ in ()).throw(AssertionError("cleanup must not run twice")),
    )

    assert module.run_child() == 0

    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"steps": 1, "error": "runtime stop", "aborted": True}]


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


def test_steps_benchmark_child_result_bounds_text_and_collections(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_result_bounds_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    long_error = "blocked:" + ("x" * (module.STEPS_RESULT_TEXT_FIELD_MAX_CHARS * 2))
    cleanup = {
        "detail": long_error,
        "events": list(range(module.STEPS_RESULT_COLLECTION_MAX_ITEMS + 20)),
        "nested": {
            f"item-{index}": long_error
            for index in range(module.STEPS_RESULT_COLLECTION_MAX_ITEMS + 20)
        },
    }
    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(module.RuntimeMemoryAbort(long_error)),
    )
    monkeypatch.setattr(module, "mlx_cleanup", lambda: cleanup)

    assert module.run_child() == 0

    records = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    record = records[0]
    assert "<truncated>" in record["error"]
    assert len(record["error"]) <= module.STEPS_RESULT_TEXT_FIELD_MAX_CHARS
    assert "<truncated>" in record["cleanup"]["detail"]
    assert len(record["cleanup"]["events"]) == module.STEPS_RESULT_COLLECTION_MAX_ITEMS + 1
    assert record["cleanup"]["events"][-1] == {"__truncated_items__": True}
    assert len(record["cleanup"]["nested"]) == module.STEPS_RESULT_COLLECTION_MAX_ITEMS + 1
    assert record["cleanup"]["nested"]["__truncated_items__"] is True


def test_steps_benchmark_summarizes_unknown_result_values_without_repr_or_str(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_unknown_result_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)
    result_path = tmp_path / "results.jsonl"

    class UnsafeValue:
        def __repr__(self):
            raise AssertionError("steps result bounding must not call repr on unknown values")

        def __str__(self):
            raise AssertionError("steps result bounding must not call str on unknown values")

    class UnsafeKey:
        def __repr__(self):
            raise AssertionError("steps result bounding must not call repr on unknown keys")

        def __str__(self):
            raise AssertionError("steps result bounding must not call str on unknown keys")

    module._write_steps_jsonl(
        result_path,
        [{"steps": 1, "cleanup": {UnsafeKey(): UnsafeValue()}}],
    )

    record = json.loads(result_path.read_text(encoding="utf-8"))
    cleanup = record["cleanup"]
    key = next(iter(cleanup))
    assert "UnsafeKey" in key
    assert "UnsafeValue" in cleanup[key]


def test_steps_benchmark_child_summarizes_exceptions_without_str_or_traceback(tmp_path, monkeypatch, capsys):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_exception_summary_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    result_path = tmp_path / "steps" / "child.json"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(result_path))
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    spec.loader.exec_module(module)

    class UnsafeArg:
        def __repr__(self):
            raise AssertionError("exception arg repr must not be called")

        def __str__(self):
            raise AssertionError("exception arg str must not be called")

    class UnsafeException(Exception):
        def __str__(self):
            raise AssertionError("exception str must not be called")

    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(UnsafeException(UnsafeArg())),
    )
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {})

    assert module.run_child() == 0

    output = capsys.readouterr().out
    assert "traceback suppressed" in output
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert "UnsafeArg" in record["error"]


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


def test_steps_benchmark_run_single_fails_closed_when_restart_required(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_restart_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "run_counter", lambda: 1)
    monkeypatch.setattr(module, "should_restart_process", lambda: True)
    monkeypatch.setattr(
        module,
        "check_memory_guard",
        lambda label: (_ for _ in ()).throw(AssertionError("memory guard must not run after restart is required")),
    )
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(AssertionError("dependency probing must not run after restart is required")),
    )

    with pytest.raises(module.MemoryGuardError, match="process restart required after 1 consecutive MLX runs"):
        module.run_single(1)


def test_steps_benchmark_run_single_checks_dependency_before_mlx_import(tmp_path, monkeypatch):
    import builtins
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
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"mlx", "mlx.core"}:
            raise AssertionError("mlx.core must not be imported before dependency preflight")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("mlx", None)
    sys.modules.pop("mlx.core", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

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
    frame_bytes = 2 * 4 * 4 * 3 * 4
    assert host_checks == [
        (
            frame_bytes * module.VIDEO_STATS_ALLOCATION_MULTIPLIER,
            "steps_1 quality metrics",
        ),
        (
            frame_bytes * module.PNG_FRAME_ALLOCATION_MULTIPLIER,
            "steps_1 png frames",
        ),
    ]
    assert len(saved) == 2


def test_steps_benchmark_run_single_aborts_when_post_run_cleanup_fails(tmp_path, monkeypatch):
    import importlib.util
    import numpy as np
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_cleanup_failure_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "1")
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
            return np.zeros((1, 4, 4, 3), dtype=np.uint8)

    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    fake_image_module = types.SimpleNamespace(
        fromarray=lambda frame: types.SimpleNamespace(save=lambda path: None)
    )
    counter_calls: list[str] = []

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "check_host_allocation_headroom", lambda required, *, label: None)
    monkeypatch.setattr(
        module,
        "mlx_cleanup",
        lambda: {
            "mlx_loaded": True,
            "mlx_cache_cleared": False,
            "mlx_cleanup_error": "failed to clear MLX cache",
        },
    )
    monkeypatch.setattr(module, "increment_run_counter", lambda: counter_calls.append("counter") or 1)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    with pytest.raises(module.RuntimeMemoryAbort, match="MLX cleanup failed") as caught:
        module.run_single(1)

    assert module._exception_cleanup_done(caught.value) is True
    assert counter_calls == []


def test_steps_benchmark_run_single_does_not_trust_underreported_video_nbytes(tmp_path, monkeypatch):
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_video_nbytes_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class FakeVideo:
        shape = (2, 4, 4, 3)
        nbytes = 0

        def min(self):
            return 0

        def max(self):
            return 255

        def mean(self):
            return 127.0

        def std(self):
            return 1.0

        def __getitem__(self, index):
            return object()

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
            return FakeVideo()

    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    fake_image_module = types.SimpleNamespace(fromarray=lambda frame: types.SimpleNamespace(save=lambda path: None))
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
    frame_floor_bytes = 2 * 4 * 4 * 3 * 4
    assert host_checks == [
        (
            frame_floor_bytes * module.VIDEO_STATS_ALLOCATION_MULTIPLIER,
            "steps_1 quality metrics",
        ),
        (
            frame_floor_bytes * module.PNG_FRAME_ALLOCATION_MULTIPLIER,
            "steps_1 png frames",
        ),
    ]


def test_steps_benchmark_run_single_reuses_validated_video_shape(tmp_path, monkeypatch):
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_video_shape_reuse_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class SinglePassShape:
        def __init__(self, dims):
            self._dims = dims
            self._iterations = 0

        def __iter__(self):
            self._iterations += 1
            if self._iterations > 1:
                raise AssertionError("run_single must not re-read video shape after validation")
            yield from self._dims

    class FakeVideo:
        nbytes = 0

        def __init__(self):
            self.shape = SinglePassShape((2, 4, 4, 3))

        def min(self):
            return 0

        def max(self):
            return 255

        def mean(self):
            return 127.0

        def std(self):
            return 1.0

        def __getitem__(self, index):
            return object()

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
            return FakeVideo()

    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    saved: list[str] = []
    fake_image_module = types.SimpleNamespace(
        fromarray=lambda frame: types.SimpleNamespace(save=lambda path: saved.append(path))
    )

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {"freed_gb": 0, "free_after_gb": 100})
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(module, "check_host_allocation_headroom", lambda required, *, label: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    result = module.run_single(1)

    assert result["video_shape"] == [2, 4, 4, 3]
    assert len(saved) == 2


def test_steps_benchmark_pixel_metric_failure_aborts_and_cleans_up(tmp_path, monkeypatch):
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_pixel_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class UnsafeMetric:
        def __int__(self):
            raise AssertionError("unsafe metric conversion")

        def __repr__(self):
            raise AssertionError("metric guard must not repr unsafe values")

    class FakeVideo:
        shape = (2, 4, 4, 3)
        nbytes = 0

        def min(self):
            return UnsafeMetric()

        def max(self):
            return 255

        def mean(self):
            return 127.0

        def std(self):
            return 1.0

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
            return FakeVideo()

    cleanup_calls: list[str] = []
    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: cleanup_calls.append("cleanup") or {"freed_gb": 0})
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(module, "check_host_allocation_headroom", lambda required, *, label: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    with pytest.raises(module.RuntimeMemoryAbort, match="decoded video pixel metric failed"):
        module.run_single(1)

    assert cleanup_calls == ["cleanup"]


@pytest.mark.parametrize("failure", ["fromarray", "save"])
def test_steps_benchmark_png_save_failure_aborts_and_cleans_up(tmp_path, monkeypatch, failure):
    import importlib.util
    import numpy as np
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location(f"steps_benchmark_png_failure_{failure}_test", module_path)
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

    class FakeImage:
        def save(self, path):
            if failure == "save":
                raise AssertionError("png save failed")

    def fake_fromarray(frame):
        if failure == "fromarray":
            raise AssertionError("png frame materialization failed")
        return FakeImage()

    cleanup_calls: list[str] = []
    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    fake_image_module = types.SimpleNamespace(fromarray=fake_fromarray)

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: cleanup_calls.append("cleanup") or {"freed_gb": 0})
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(module, "check_host_allocation_headroom", lambda required, *, label: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    with pytest.raises(module.RuntimeMemoryAbort, match="PNG frame save failed"):
        module.run_single(1)

    assert cleanup_calls == ["cleanup"]


def test_steps_benchmark_shape_validation_rejects_unsafe_dimensions_without_repr(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_shape_dim_repr_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)

    class UnsafeDim:
        def __repr__(self):
            raise AssertionError("steps shape guard must not call repr on unknown dimensions")

    class FakeVideo:
        shape = (2, 4, 4, UnsafeDim())

    with pytest.raises(module.RuntimeMemoryAbort, match="UnsafeDim"):
        module._check_decoded_video_shape(FakeVideo(), label="quality metrics")


def test_steps_benchmark_mlx_eval_failure_aborts_and_cleans_up(tmp_path, monkeypatch):
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_eval_abort_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    class FakePipeline:
        def load_model(self):
            return {}

        def prepare_prompt(self, *, prompt, negative_prompt):
            return {"prompt": prompt, "negative_prompt": negative_prompt}

        def encode_text(self, prepared):
            return object()

    cleanup_calls: list[str] = []
    fake_mx = types.SimpleNamespace(
        eval=lambda *args: (_ for _ in ()).throw(RuntimeError("metal sync failed")),
    )

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: cleanup_calls.append("cleanup") or {"freed_gb": 0})
    monkeypatch.setattr(module, "increment_run_counter", lambda: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    with pytest.raises(module.RuntimeMemoryAbort, match="MLX eval failed") as caught:
        module.run_single(1)

    assert cleanup_calls == ["cleanup"]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_steps_benchmark_eval_failure_clears_cause_traceback_before_cleanup(tmp_path, monkeypatch):
    import gc
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_eval_cause_release_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)

    class HeavyLocal:
        pass

    ref: weakref.ReferenceType[object] | None = None
    class FakeMx:
        def eval(self, target):
            nonlocal ref
            heavy = HeavyLocal()
            ref = weakref.ref(heavy)
            raise RuntimeError("metal eval failed")

    cleanup_calls: list[str] = []

    def cleanup():
        gc.collect()
        assert ref is not None
        assert ref() is None
        cleanup_calls.append("cleanup")
        return {"freed_gb": 0}

    monkeypatch.setattr(module, "mlx_cleanup", cleanup)

    with pytest.raises(module.RuntimeMemoryAbort, match="MLX eval failed") as caught:
        module._eval_mlx(FakeMx(), object(), label="eval cause")

    assert cleanup_calls == ["cleanup"]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_steps_benchmark_eval_failure_fails_closed_when_cleanup_fails(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_eval_cleanup_failure_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)

    class FakeMx:
        def eval(self, target):
            raise RuntimeError("metal eval failed")

    cleanup_errors: list[RuntimeError] = []

    def cleanup():
        exc = RuntimeError("cleanup failed")
        cleanup_errors.append(exc)
        raise exc

    monkeypatch.setattr(module, "mlx_cleanup", cleanup)

    with pytest.raises(module.RuntimeMemoryAbort, match="MLX eval failed") as caught:
        module._eval_mlx(FakeMx(), object(), label="eval cleanup")

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert cleanup_errors
    assert cleanup_errors[0].__traceback__ is None
    assert cleanup_errors[0].__cause__ is None
    assert cleanup_errors[0].__context__ is None


def test_steps_benchmark_child_cleanup_clears_traceback_locals_before_mlx_cleanup(tmp_path, monkeypatch):
    import gc
    import importlib.util
    import fastgen_profiler.backends.ltx23_mlx_adapter as ltx_adapter

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_traceback_release_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    output_base = tmp_path / "steps"
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(output_base))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_STEP", "1")
    monkeypatch.setenv("FASTGEN_STEPS_CHILD_RESULT", str(output_base / "child.json"))
    spec.loader.exec_module(module)

    class HeavyLocal:
        pass

    refs: dict[str, weakref.ReferenceType[object]] = {}

    def tracked(name: str):
        value = HeavyLocal()
        refs[name] = weakref.ref(value)
        return value

    class FakePipeline:
        def __init__(self):
            refs["pipe"] = weakref.ref(self)

        def load_model(self):
            return {}

        def prepare_prompt(self, *, prompt, negative_prompt):
            return tracked("prepared")

        def encode_text(self, prepared):
            return tracked("context")

        def init_latents(self, *, seed, width, height, frames):
            return tracked("latents")

        def denoise_step(self, latents, *, step_index, steps, guidance, cache):
            return latents

        def decode(self, latents):
            raise RuntimeError("decode failed after runtime opened")

    fake_mx = types.SimpleNamespace(eval=lambda *args: None, array=lambda value: value)
    cleanup_calls: list[str] = []

    def cleanup():
        gc.collect()
        assert refs
        assert {name: ref() for name, ref in refs.items()} == {name: None for name in refs}
        cleanup_calls.append("cleanup")
        return {"freed_gb": 0}

    monkeypatch.setattr(module, "check_memory_guard", lambda label: {"free_gb": 100})
    monkeypatch.setattr(module, "check_run_allocation_budget", lambda **kwargs: {"shape_floor_gb": 1})
    monkeypatch.setattr(module, "check_runtime_memory", lambda label: None)
    monkeypatch.setattr(module, "mlx_cleanup", cleanup)
    monkeypatch.setattr(module, "increment_run_counter", lambda: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ltx_adapter, "create_ltx23_pipeline", lambda **kwargs: FakePipeline())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    assert module.run_child() == 0

    assert cleanup_calls == ["cleanup"]


def test_steps_benchmark_generic_failure_consumes_process_slot(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_generic_failure_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1")
    spec.loader.exec_module(module)

    calls: list[str] = []
    monkeypatch.setattr(module, "run_counter", lambda: 0)
    monkeypatch.setattr(module, "should_restart_process", lambda: False)
    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(RuntimeError("metal state unknown")),
    )
    monkeypatch.setattr(module, "increment_run_counter", lambda: calls.append("counter") or 1)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: calls.append("cleanup") or {"freed_gb": 0})

    assert module.main() == 1

    assert calls == ["counter", "cleanup"]
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"steps": 1, "error": "metal state unknown"}]


def test_steps_benchmark_generic_failure_does_not_stringify_unsafe_exception(tmp_path, monkeypatch, capsys):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_unsafe_exception_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1")
    spec.loader.exec_module(module)

    class UnsafeError(Exception):
        def __str__(self):
            raise AssertionError("steps benchmark must not call str on exceptions")

    monkeypatch.setattr(module, "run_counter", lambda: 0)
    monkeypatch.setattr(module, "should_restart_process", lambda: False)
    monkeypatch.setattr(
        module,
        "run_single",
        lambda steps: (_ for _ in ()).throw(UnsafeError("metal state unknown")),
    )
    monkeypatch.setattr(module, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(module, "mlx_cleanup", lambda: {"freed_gb": 0})

    assert module.main() == 1

    captured = capsys.readouterr()
    assert "traceback suppressed" in captured.out
    records = [
        json.loads(line)
        for line in module.RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["steps"] == 1
    assert records[0]["error"] == "metal state unknown"


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

    with pytest.raises(module.RuntimeMemoryAbort, match="decoded benchmark video shape rank is 3, expected 4"):
        module.run_single(1)


def test_steps_benchmark_rejects_unbounded_video_shape_metadata(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_unbounded_shape_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class Shape:
        def __iter__(self):
            yield from (2, 4, 4, 3, 1)
            raise AssertionError("shape guard must not consume beyond expected rank")

    class FakeVideo:
        shape = Shape()

    with pytest.raises(module.RuntimeMemoryAbort, match="shape rank"):
        module._check_decoded_video_shape(FakeVideo(), label="steps_1")


def test_steps_benchmark_rejects_non_iterable_video_shape_without_exception_chain(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_non_iterable_shape_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)

    class FakeVideo:
        shape = object()

    with pytest.raises(module.RuntimeMemoryAbort, match="shape is not iterable") as caught:
        module._check_decoded_video_shape(FakeVideo(), label="steps_1")

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_steps_benchmark_rejects_short_video_shape_metadata(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_short_shape_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_WIDTH", "4")
    monkeypatch.setenv("FASTGEN_STEPS_HEIGHT", "4")
    monkeypatch.setenv("FASTGEN_STEPS_FRAMES", "2")
    spec.loader.exec_module(module)

    class FakeVideo:
        shape = (2, 4, 4)

    with pytest.raises(module.RuntimeMemoryAbort, match="shape rank is 3, expected 4"):
        module._check_decoded_video_shape(FakeVideo(), label="steps_1")


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

    child_log = module.OUTPUT_BASE / "steps_1.child.log"
    assert result == {
        "steps": 1,
        "error": "child process exited 7 without a result record",
        "aborted": True,
        "log_path": str(child_log),
    }
    assert not child_log.exists()
    assert not stale.exists()


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
    assert not child_result.exists()
    assert not (module.OUTPUT_BASE / "steps_1.child.log").exists()


def test_steps_benchmark_streams_child_result_without_read_text(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_stream_child_result_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    spec.loader.exec_module(module)
    module.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    child_result = module.OUTPUT_BASE / "steps_1.child.json"

    def fake_run(*args, **_kwargs):
        child_result.write_text(
            json.dumps({"steps": 1, "progress": "ignored"}) + "\n"
            + json.dumps({"steps": 1, "denoise_total_s": 1, "vae_decode_s": 2}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    with patch.object(module.Path, "read_text", side_effect=AssertionError("child result must be streamed")):
        with patch.object(module.subprocess, "run", side_effect=fake_run):
            result = module.run_step_in_child(1)

    assert result["steps"] == 1
    assert result["denoise_total_s"] == 1
    assert result["vae_decode_s"] == 2
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
    assert not child_result.exists()
    assert not (module.OUTPUT_BASE / "steps_1.child.log").exists()


def test_steps_benchmark_rejects_unbounded_parent_result_stream(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_parent_writer_limit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)

    def records():
        while True:
            yield {"steps": 1, "skipped": True}

    with pytest.raises(module.MemoryGuardError, match="steps result record limit exceeded"):
        module._write_steps_jsonl(tmp_path / "results.jsonl", records(), max_records=1)

    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines() == [
        json.dumps({"steps": 1, "skipped": True})
    ]


def test_steps_benchmark_rejects_oversized_parent_result_record(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_parent_record_byte_limit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    spec.loader.exec_module(module)

    result_path = tmp_path / "results.jsonl"
    with pytest.raises(module.MemoryGuardError, match="steps result record byte limit exceeded"):
        module._write_steps_jsonl(
            result_path,
            [{"steps": 1, "error": "x" * 100}],
            max_record_bytes=10,
        )

    assert not result_path.exists() or result_path.read_text(encoding="utf-8") == ""


def test_steps_benchmark_child_result_write_rejects_oversized_record(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_child_record_byte_limit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_RESULT_RECORD_MAX_BYTES", "10")
    spec.loader.exec_module(module)
    module.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    result_path = module.OUTPUT_BASE / "steps_1.child.json"
    with pytest.raises(module.MemoryGuardError, match="steps result record byte limit exceeded"):
        module._write_child_result_file(result_path, {"steps": 1, "error": "x" * 100})

    assert not result_path.exists()
    assert not result_path.with_suffix(result_path.suffix + ".tmp").exists()


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


def test_steps_benchmark_parent_recovery_does_not_import_or_cleanup_mlx(tmp_path, monkeypatch):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "steps_benchmark.py"
    spec = importlib.util.spec_from_file_location("steps_benchmark_parent_recovery_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("FASTGEN_STEPS_OUTPUT_BASE", str(tmp_path / "steps"))
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY", "1")
    monkeypatch.setenv("FASTGEN_STEPS_VALUES", "1,2")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_counter", lambda: 0)

    launched: list[int] = []
    guard_labels: list[str] = []
    sleeps: list[float] = []

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

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"mlx", "mlx.core"}:
            raise AssertionError("parent recovery must not initialize MLX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(module, "run_step_in_child", fake_child)
    monkeypatch.setattr(module, "check_memory_guard", lambda label: guard_labels.append(label) or {"free_gb": 100})
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        module,
        "mlx_cleanup",
        lambda: (_ for _ in ()).throw(AssertionError("parent recovery must not cleanup MLX")),
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert module.main() == 0

    assert launched == [1, 2]
    assert guard_labels == ["pre-steps_2"]
    assert sleeps == [module.COOLDOWN_SECONDS]


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
