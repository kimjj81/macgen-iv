"""Tests for mlx_guard: system snapshot, adaptive batch, runtime watchdog."""

from __future__ import annotations

import builtins
import json
import sys
import types

import pytest
from unittest.mock import patch, MagicMock

from fastgen_profiler.mlx_guard import (
    SystemSnapshot,
    AdaptiveBatchConfig,
    AdaptiveBatchManager,
    BatchDecision,
    MemoryGuardError,
    RuntimeMemoryAbort,
    check_memory_guard,
    check_host_allocation_headroom,
    check_run_allocation_budget,
    check_runtime_memory,
    check_text_prompt_budget,
    check_token_sequence_budget,
    configure_mlx_resource_limits,
    estimate_video_run_floor_bytes,
    adaptive_batch_config_from_run,
    free_memory_bytes,
    increment_run_counter,
    inter_run_recovery,
    inter_run_system_recovery,
    mlx_cleanup,
    reset_run_counter,
    run_counter,
    should_restart_process,
)


def _install_fake_mlx(monkeypatch):
    fake_mx = types.SimpleNamespace(
        set_memory_limit=lambda value: 0,
        set_cache_limit=lambda value: 0,
        set_wired_limit=lambda value: 0,
        clear_cache=lambda: None,
        eval=lambda *args: None,
        array=lambda value: value,
        get_active_memory=lambda: 0,
        get_cache_memory=lambda: 0,
        get_peak_memory=lambda: 0,
    )
    fake_mlx = types.SimpleNamespace(core=fake_mx)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return fake_mx


def _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir, *, max_tokens=16):
    text_encoder_dir.mkdir(exist_ok=True)
    tokenizer_dir.mkdir(exist_ok=True)
    (text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "gemma3",
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "intermediate_size": 32,
                    "num_attention_heads": 1,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 32,
                    "num_key_value_heads": 1,
                    "rope_theta": 10000,
                    "query_pre_attn_scalar": 1.0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "max_position_embeddings": max_tokens,
                }
            }
        ),
        encoding="utf-8",
    )
    (text_encoder_dir / "model.safetensors").write_bytes(b"x")
    (tokenizer_dir / "tokenizer.json").write_bytes(b"x")


def _install_fake_transformers_tokenizer(monkeypatch, *, token_count=2):
    class FakeTokenizer:
        def __call__(self, prompt, return_tensors):
            return {"input_ids": [list(range(token_count))]}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, local_files_only):
            return FakeTokenizer()

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)


# ---------------------------------------------------------------------------
# SystemSnapshot
# ---------------------------------------------------------------------------

class TestSystemSnapshot:
    def test_free_gb_known(self):
        snap = SystemSnapshot(
            free_bytes=8 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.3,
            swap_files=2,
            free_fraction=8 / 128,
        )
        assert "8.6" in snap.free_gb()  # 8 GiB = 8.589... GB (decimal)
        assert "free=" in snap.summary()
        assert "pressure=30%" in snap.summary()
        assert "swap=2" in snap.summary()

    def test_free_gb_unknown(self):
        snap = SystemSnapshot(
            free_bytes=None,
            total_bytes=None,
            pressure=None,
            swap_files=None,
            free_fraction=None,
        )
        assert snap.free_gb() == "?"
        assert snap.summary() == "free=?GB"


class TestVmStatParsing:
    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_free_memory_parses_commas_and_page_size(self, mock_run):
        import subprocess

        def fake_run(args, **kwargs):
            kwargs["stdout"].write(
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                "Pages free:                               1,000.\n"
                "Pages active:                             3,000.\n"
                "Pages inactive:                           2,000.\n"
                .encode("utf-8")
            )
            return subprocess.CompletedProcess(args=args, returncode=0)

        mock_run.side_effect = fake_run

        assert free_memory_bytes() == 3000 * 4096
        kwargs = mock_run.call_args.kwargs
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert "capture_output" not in kwargs

    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_free_memory_treats_oversized_telemetry_as_unknown(self, mock_run):
        import subprocess

        def fake_run(args, **kwargs):
            kwargs["stdout"].write(b"x" * 65_537)
            return subprocess.CompletedProcess(args=args, returncode=0)

        mock_run.side_effect = fake_run

        assert free_memory_bytes() is None


class TestSystemTelemetryFailures:
    def test_swap_file_count_stops_after_guard_threshold(self, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        consumed = 0

        class FakeEntry:
            def __init__(self, name):
                self.name = name

        class FakeSwapDir:
            def __init__(self, path):
                self.path = path

            def is_dir(self):
                return True

            def iterdir(self):
                nonlocal consumed
                for index in range(mlx_guard.MAX_SWAP_FILES + 100):
                    consumed += 1
                    if consumed > mlx_guard.MAX_SWAP_FILES + 1:
                        raise AssertionError("swap scan must stop after threshold is exceeded")
                    yield FakeEntry(f"swapfile{index}")

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        monkeypatch.setattr(mlx_guard, "Path", FakeSwapDir)

        assert mlx_guard.swap_file_count() == mlx_guard.MAX_SWAP_FILES + 1
        assert consumed == mlx_guard.MAX_SWAP_FILES + 1

    def test_system_snapshot_tolerates_telemetry_helper_exceptions(self, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard, "free_memory_bytes", lambda: (_ for _ in ()).throw(RuntimeError("free failed")))
        monkeypatch.setattr(mlx_guard, "total_memory_bytes", lambda: (_ for _ in ()).throw(RuntimeError("total failed")))
        monkeypatch.setattr(mlx_guard, "memory_pressure_fraction", lambda: (_ for _ in ()).throw(RuntimeError("pressure failed")))
        monkeypatch.setattr(mlx_guard, "swap_file_count", lambda: (_ for _ in ()).throw(RuntimeError("swap failed")))

        snap = mlx_guard.system_snapshot()

        assert snap == SystemSnapshot(
            free_bytes=None,
            total_bytes=None,
            pressure=None,
            swap_files=None,
            free_fraction=None,
        )

    def test_memory_guard_fails_closed_after_snapshot_telemetry_exceptions(self, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        monkeypatch.setattr(mlx_guard, "free_memory_bytes", lambda: (_ for _ in ()).throw(RuntimeError("free failed")))
        monkeypatch.setattr(mlx_guard, "total_memory_bytes", lambda: 128 * 1024 ** 3)
        monkeypatch.setattr(mlx_guard, "memory_pressure_fraction", lambda: 0.1)
        monkeypatch.setattr(mlx_guard, "swap_file_count", lambda: 0)

        with pytest.raises(MemoryGuardError, match="cannot read vm_stat free memory"):
            check_memory_guard(label="telemetry-exception")


# ---------------------------------------------------------------------------
# Pre-run guard (Guard 1)
# ---------------------------------------------------------------------------

class TestCheckMemoryGuard:
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_passes_with_enough_memory(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=10 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=3,
            free_fraction=10 / 128,
        )
        status = check_memory_guard(label="test")
        assert status["label"] == "test"
        assert status["free_gb"] is not None and status["free_gb"] > 9

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_on_low_memory(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=1 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.9,
            swap_files=15,
            free_fraction=1 / 128,
        )
        with pytest.raises(MemoryGuardError, match=r"only \d+\.\dGB free"):
            check_memory_guard(label="low-mem")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_on_high_pressure(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=10 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.96,
            swap_files=3,
            free_fraction=10 / 128,
        )
        with pytest.raises(MemoryGuardError, match="pressure at 96"):
            check_memory_guard(label="high-pressure")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_on_too_many_swap_files(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=10 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.3,
            swap_files=25,
            free_fraction=10 / 128,
        )
        with pytest.raises(MemoryGuardError, match="25 swap files"):
            check_memory_guard(label="swap-heavy")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_fails_closed_when_telemetry_unavailable(self, mock_snap, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=None,
            total_bytes=None,
            pressure=None,
            swap_files=None,
            free_fraction=None,
        )

        with pytest.raises(MemoryGuardError, match="cannot read vm_stat"):
            check_memory_guard(label="blind")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_fails_closed_when_free_memory_unavailable_even_with_pressure(self, mock_snap, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=None,
            total_bytes=128 * 1024 ** 3,
            pressure=0.1,
            swap_files=0,
            free_fraction=None,
        )

        with pytest.raises(MemoryGuardError, match="cannot read vm_stat free memory"):
            check_memory_guard(label="blind-free")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_fails_closed_when_swap_telemetry_unavailable(self, mock_snap, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.1,
            swap_files=None,
            free_fraction=80 / 128,
        )

        with pytest.raises(MemoryGuardError, match="swap file state"):
            check_memory_guard(label="blind-swap")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_fails_closed_when_pressure_telemetry_unavailable(self, mock_snap, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=None,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(MemoryGuardError, match="memory pressure"):
            check_memory_guard(label="blind-pressure")

    @pytest.mark.parametrize(
        "snapshot",
        [
            SystemSnapshot(
                free_bytes=-1,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=0,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=129 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=-0.1,
                swap_files=0,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=1.1,
                swap_files=0,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=-1,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=1.1,
            ),
        ],
    )
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_fails_closed_on_invalid_memory_telemetry(self, mock_snap, snapshot):
        mock_snap.return_value = snapshot

        with pytest.raises(MemoryGuardError, match="invalid memory telemetry"):
            check_memory_guard(label="invalid-telemetry")


# ---------------------------------------------------------------------------
# Runtime watchdog (Guard 3)
# ---------------------------------------------------------------------------

class TestCheckRuntimeMemory:
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_returns_snapshot_when_ok(self, mock_snap):
        snap = SystemSnapshot(
            free_bytes=10 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.5,
            swap_files=3,
            free_fraction=10 / 128,
        )
        mock_snap.return_value = snap
        result = check_runtime_memory(label="step-ok")
        assert result is snap

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_on_low_free_memory(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=1 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.7,
            swap_files=10,
            free_fraction=1 / 128,
        )
        with pytest.raises(RuntimeMemoryAbort, match=r"only \d+\.\dGB free"):
            check_runtime_memory(label="step-low")

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_runtime_abort_runs_cleanup_first(self, mock_snap, mock_cleanup):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=1 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.7,
            swap_files=10,
            free_fraction=1 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort):
            check_runtime_memory(label="step-low")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_runtime_fails_closed_when_system_snapshot_raises(self, mock_snap, mock_cleanup):
        mock_snap.side_effect = RuntimeError("telemetry command failed")

        with pytest.raises(RuntimeMemoryAbort, match="cannot capture system memory telemetry"):
            check_runtime_memory(label="snapshot-error")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_on_high_pressure(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=5 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.95,
            swap_files=10,
            free_fraction=5 / 128,
        )
        with pytest.raises(RuntimeMemoryAbort, match="pressure at 95"):
            check_runtime_memory(label="step-pressure")

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_raises_when_mlx_memory_nears_configured_limit(self, mock_snap, mock_cleanup, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(mx, "get_active_memory", lambda: 900)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: 30)
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", 1000)
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="MLX active\\+cache memory"):
            check_runtime_memory(label="mlx-limit")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_mlx_memory_below_configured_limit_passes(self, mock_snap, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(mx, "get_active_memory", lambda: 500)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: 100)
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", 1000)
        snap = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=80 / 128,
        )
        mock_snap.return_value = snap

        assert check_runtime_memory(label="mlx-ok") is snap

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_runtime_fails_closed_when_memory_telemetry_unavailable(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=None,
            total_bytes=128 * 1024 ** 3,
            pressure=None,
            swap_files=0,
            free_fraction=None,
        )

        with pytest.raises(RuntimeMemoryAbort, match="free memory telemetry is unavailable"):
            check_runtime_memory(label="telemetry-missing")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_runtime_fails_closed_when_free_memory_unavailable_even_with_pressure(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=None,
            total_bytes=128 * 1024 ** 3,
            pressure=0.1,
            swap_files=0,
            free_fraction=None,
        )

        with pytest.raises(RuntimeMemoryAbort, match="free memory telemetry is unavailable"):
            check_runtime_memory(label="runtime-blind-free")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_runtime_fails_closed_when_swap_telemetry_unavailable(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.1,
            swap_files=None,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="swap telemetry is unavailable"):
            check_runtime_memory(label="runtime-blind-swap")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_darwin_runtime_fails_closed_when_pressure_telemetry_unavailable(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=None,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="pressure telemetry is unavailable"):
            check_runtime_memory(label="runtime-blind-pressure")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_runtime_fails_closed_when_mlx_allocator_telemetry_fails(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(mx, "get_active_memory", lambda: (_ for _ in ()).throw(RuntimeError("no counter")))
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", 1000)
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="cannot read MLX allocator memory"):
            check_runtime_memory(label="mlx-counter-missing")

        mock_cleanup.assert_called_once()

    @pytest.mark.parametrize("counter_value", [-1, "100", 1.5, True])
    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_runtime_fails_closed_when_mlx_allocator_counter_is_invalid(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
        counter_value,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(mx, "get_active_memory", lambda: counter_value)
        monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", 1000)
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="cannot read MLX allocator memory"):
            check_runtime_memory(label="mlx-invalid-counter")

        mock_cleanup.assert_called_once()

    @pytest.mark.parametrize(
        "snapshot",
        [
            SystemSnapshot(
                free_bytes=-1,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=0,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=129 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=None,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=-0.1,
                swap_files=0,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=1.1,
                swap_files=0,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=-1,
                free_fraction=80 / 128,
            ),
            SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.1,
                swap_files=0,
                free_fraction=1.1,
            ),
        ],
    )
    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_runtime_fails_closed_on_invalid_memory_telemetry(
        self,
        mock_snap,
        mock_cleanup,
        snapshot,
    ):
        mock_snap.return_value = snapshot

        with pytest.raises(RuntimeMemoryAbort, match="invalid memory telemetry"):
            check_runtime_memory(label="runtime-invalid-telemetry")

        mock_cleanup.assert_called_once()

    def test_cleanup_does_not_import_mlx_when_runtime_is_absent(self, monkeypatch):
        sys.modules.pop("mlx.core", None)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("cleanup must not initialize MLX")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        status = mlx_cleanup()

        assert "free_before_gb" in status
        assert status["mlx_loaded"] is False
        assert status["mlx_cache_cleared"] is False

    def test_cleanup_reports_mlx_cache_clear_status(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)
        calls: list[str] = []
        monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear"))
        monkeypatch.setattr(mx, "eval", lambda *args: calls.append("eval"))
        monkeypatch.setattr(mx, "array", lambda value: (_ for _ in ()).throw(AssertionError("cleanup must not allocate")))

        status = mlx_cleanup()

        assert calls == ["clear"]
        assert status["mlx_loaded"] is True
        assert status["mlx_cache_cleared"] is True
        assert status["mlx_cleanup_error"] is None

    def test_cleanup_reports_mlx_cache_clear_failure(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(mx, "clear_cache", lambda: (_ for _ in ()).throw(RuntimeError("cache stuck")))

        status = mlx_cleanup()

        assert status["mlx_loaded"] is True
        assert status["mlx_cache_cleared"] is False
        assert status["mlx_cleanup_error"] == "failed to clear MLX cache"

    def test_cleanup_records_free_memory_telemetry_failure(self, monkeypatch):
        calls = 0

        def failing_free_memory():
            nonlocal calls
            calls += 1
            raise RuntimeError("vm_stat failed")

        monkeypatch.setattr("fastgen_profiler.mlx_guard.free_memory_bytes", failing_free_memory)

        status = mlx_cleanup()

        assert calls == 2
        assert status["free_before_gb"] is None
        assert status["free_after_gb"] is None
        assert status["freed_gb"] is None
        assert status["memory_telemetry_error"] == "failed to read free memory before cleanup"


class TestConfigureMlxResourceLimits:
    def test_sets_conservative_default_limits(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        snap = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=80 / 128,
        )

        status = configure_mlx_resource_limits(snapshot=snap, label="limits")

        assert calls["memory"] == 72 * 1024 ** 3
        assert calls["cache"] == 1 * 1024 ** 3
        assert calls["wired"] == calls["memory"]
        assert status["memory_limit_gb"] is not None

    def test_env_overrides_limits(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "12")
        monkeypatch.setenv("FASTGEN_MLX_CACHE_LIMIT_GB", "0.5")
        monkeypatch.setenv("FASTGEN_MLX_WIRED_LIMIT_GB", "10")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=None,
            ),
            label="limits",
        )

        assert calls["memory"] == 12 * 1024 ** 3
        assert calls["cache"] == int(0.5 * 1024 ** 3)
        assert calls["wired"] == 10 * 1024 ** 3

    def test_limit_set_failure_runs_cleanup_after_mlx_import(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: (_ for _ in ()).throw(RuntimeError("limit failed")))
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        with pytest.raises(MemoryGuardError, match="failed to set MLX memory limits"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="limit-fail",
            )

        assert cleanup_calls == ["cleanup"]

    def test_wired_limit_set_failure_fails_closed_after_mlx_import(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_WIRED_LIMIT_GB", "10")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: (_ for _ in ()).throw(RuntimeError("wired failed")))
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        with pytest.raises(MemoryGuardError, match="failed to set MLX wired memory limit"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="wired-fail",
            )

        assert cleanup_calls == ["cleanup"]

    def test_mlx_import_failure_runs_cleanup(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx.core", None)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise RuntimeError("mlx import failed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(MemoryGuardError, match="cannot import mlx.core"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="import-fail",
            )

        assert cleanup_calls == ["cleanup"]

    @pytest.mark.parametrize("env_value", ["nan", "inf", "-inf"])
    def test_env_overrides_reject_non_finite_numbers(self, monkeypatch, env_value):
        _install_fake_mlx(monkeypatch)
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", env_value)

        with pytest.raises(MemoryGuardError, match="finite number of GB greater than zero"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="bad-env",
            )

    def test_system_reserve_rejects_too_small_positive_value(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_SYSTEM_RESERVE_GB", "1e-20")

        with pytest.raises(MemoryGuardError, match="too small"):
            check_host_allocation_headroom(1, label="tiny-reserve")

    def test_default_cache_limit_is_clamped_to_memory_limit(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "0.75")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=None,
            ),
            label="limits",
        )

        expected = int(0.75 * 1024 ** 3)
        assert calls["memory"] == expected
        assert calls["cache"] == expected
        assert calls["wired"] == expected

    def test_env_overrides_cannot_exceed_default_safe_limit(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "120")
        monkeypatch.setenv("FASTGEN_MLX_CACHE_LIMIT_GB", "120")
        monkeypatch.setenv("FASTGEN_MLX_WIRED_LIMIT_GB", "120")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=80 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=None,
            ),
            label="limits",
        )

        expected = 72 * 1024 ** 3
        assert calls["memory"] == expected
        assert calls["cache"] == expected
        assert calls["wired"] == expected

    def test_memory_limit_is_clamped_to_current_free_headroom(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "64")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=20 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=20 / 128,
            ),
            label="free-clamp",
        )

        expected = 12 * 1024 ** 3
        assert calls["memory"] == expected
        assert calls["cache"] == 1 * 1024 ** 3
        assert calls["wired"] == expected

    def test_memory_limit_fails_when_free_headroom_cannot_preserve_reserve(self, monkeypatch):
        _install_fake_mlx(monkeypatch)
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")

        with pytest.raises(MemoryGuardError, match="below minimum"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=8 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=8 / 128,
                ),
                label="free-tight",
            )

    def test_darwin_free_memory_required_for_limit_derivation(self, monkeypatch):
        _install_fake_mlx(monkeypatch)
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")

        with pytest.raises(MemoryGuardError, match="cannot read vm_stat free memory"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=None,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="free-missing",
            )

    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_probe_mlx_import_does_not_capture_child_output(self, mock_run, monkeypatch):
        import subprocess

        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.delenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", raising=False)
        mock_run.return_value = subprocess.CompletedProcess(args=["probe"], returncode=0)

        mlx_guard._probe_mlx_import("probe")

        kwargs = mock_run.call_args.kwargs
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert "capture_output" not in kwargs

    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_probe_mlx_import_failure_reports_exit_without_output_capture(self, mock_run, monkeypatch):
        import subprocess

        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.delenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", raising=False)
        mock_run.return_value = subprocess.CompletedProcess(args=["probe"], returncode=7)

        with pytest.raises(MemoryGuardError, match=r"probe exit 7"):
            mlx_guard._probe_mlx_import("probe-fail")

    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_probe_skip_env_is_test_only(self, mock_run, monkeypatch):
        import subprocess

        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "spoofed test name")
        monkeypatch.delitem(sys.modules, "pytest", raising=False)
        mock_run.return_value = subprocess.CompletedProcess(args=["probe"], returncode=0)

        mlx_guard._probe_mlx_import("not-pytest")

        mock_run.assert_called_once()

    def test_system_reserve_env_cannot_lower_default_safe_limit(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_SYSTEM_RESERVE_GB", "1")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=16 * 1024 ** 3,
                total_bytes=16 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=None,
            ),
            label="limits",
        )

        assert calls["memory"] == 8 * 1024 ** 3

    def test_system_reserve_env_can_raise_default_safe_limit(self, monkeypatch):
        mx = _install_fake_mlx(monkeypatch)

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_SYSTEM_RESERVE_GB", "120")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=128 * 1024 ** 3,
                total_bytes=128 * 1024 ** 3,
                pressure=0.2,
                swap_files=0,
                free_fraction=None,
            ),
            label="limits",
        )

        assert calls["memory"] == 8 * 1024 ** 3
        assert calls["cache"] == 1 * 1024 ** 3
        assert calls["wired"] == calls["memory"]

    def test_darwin_requires_memory_limit_when_total_memory_unknown(self, monkeypatch):
        _install_fake_mlx(monkeypatch)
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")

        with pytest.raises(MemoryGuardError, match="cannot derive MLX memory limit"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=None,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="unknown-total",
            )

    def test_darwin_explicit_memory_limit_cannot_bypass_unknown_total_memory(self, monkeypatch):
        _install_fake_mlx(monkeypatch)
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "4")

        with pytest.raises(MemoryGuardError, match="Refusing to trust explicit allocator overrides"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=None,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=None,
                ),
                label="unknown-total-env",
            )

    def test_low_total_memory_does_not_floor_to_unsafe_limit(self, monkeypatch):
        _install_fake_mlx(monkeypatch)
        monkeypatch.setenv("FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE", "1")

        with pytest.raises(MemoryGuardError, match="before importing MLX/Metal"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=4 * 1024 ** 3,
                    total_bytes=8 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=0.5,
                ),
                label="low-total",
            )

    @patch("fastgen_profiler.mlx_guard.subprocess.run")
    def test_probe_failure_blocks_before_main_process_import(self, mock_run):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=1,
            stdout="",
            stderr="RuntimeError: no metal device",
        )

        with pytest.raises(MemoryGuardError, match="MLX/Metal is unavailable"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=80 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=80 / 128,
                ),
                label="probe",
            )

    def test_low_free_memory_blocks_before_mlx_probe_or_import(self, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard, "_probe_mlx_import", lambda label: (_ for _ in ()).throw(
            AssertionError("MLX import probe must not run when pre-run memory is unsafe")
        ))

        with pytest.raises(MemoryGuardError, match="before importing MLX/Metal"):
            configure_mlx_resource_limits(
                snapshot=SystemSnapshot(
                    free_bytes=4 * 1024 ** 3,
                    total_bytes=128 * 1024 ** 3,
                    pressure=0.2,
                    swap_files=0,
                    free_fraction=4 / 128,
                ),
                label="low-free-pre-probe",
            )


class TestAllocationBudget:
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_abort_when_reserve_would_be_consumed(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=9 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=9 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="host allocation"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_reserve_env_cannot_lower_default(self, mock_snap, monkeypatch):
        monkeypatch.setenv("FASTGEN_SYSTEM_RESERVE_GB", "1")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=9 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=9 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="host allocation"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

    @pytest.mark.parametrize("required_bytes", [0, -1, 1.5, True, "1024"])
    def test_host_allocation_rejects_invalid_required_bytes(self, required_bytes):
        with pytest.raises(MemoryGuardError, match="required_bytes must be a positive integer"):
            check_host_allocation_headroom(required_bytes, label="numpy")  # type: ignore[arg-type]

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_fails_closed_when_swap_telemetry_missing(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=None,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="swap file state"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_fails_closed_when_pressure_telemetry_missing(
        self,
        mock_snap,
        mock_cleanup,
        monkeypatch,
    ):
        import fastgen_profiler.mlx_guard as mlx_guard

        monkeypatch.setattr(mlx_guard.sys, "platform", "darwin")
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=None,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="memory pressure"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_aborts_when_pressure_is_critical(self, mock_snap, mock_cleanup):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.95,
            swap_files=0,
            free_fraction=80 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="pressure at 95"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

        mock_cleanup.assert_called_once()

    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_host_allocation_fails_closed_on_invalid_memory_telemetry(self, mock_snap, mock_cleanup):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=129 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=None,
        )

        with pytest.raises(RuntimeMemoryAbort, match="invalid memory telemetry"):
            check_host_allocation_headroom(2 * 1024 ** 3, label="numpy")

        mock_cleanup.assert_called_once()

    def test_video_run_floor_scales_with_shape(self):
        small = estimate_video_run_floor_bytes(width=256, height=256, frames=4, guidance=1.0)
        large = estimate_video_run_floor_bytes(width=1024, height=1024, frames=16, guidance=3.5)
        assert large > small

    def test_video_run_floor_uses_ceil_latent_grid_for_unaligned_shape(self):
        aligned = estimate_video_run_floor_bytes(width=256, height=256, frames=4, guidance=1.0)
        unaligned = estimate_video_run_floor_bytes(width=257, height=257, frames=4, guidance=1.0)

        assert unaligned > aligned

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"width": 0, "height": 256, "frames": 4, "guidance": 1.0}, "width must be a positive integer"),
            ({"width": 256, "height": -1, "frames": 4, "guidance": 1.0}, "height must be a positive integer"),
            ({"width": 256, "height": 256, "frames": False, "guidance": 1.0}, "frames must be a positive integer"),
            ({"width": 256, "height": 256, "frames": 4, "guidance": float("nan")}, "guidance must be a finite number"),
            ({"width": 256, "height": 256, "frames": 4, "guidance": "bad"}, "guidance must be a finite number"),
        ],
    )
    def test_video_run_floor_rejects_invalid_shape_budget_inputs(self, kwargs, message):
        with pytest.raises(MemoryGuardError, match=message):
            estimate_video_run_floor_bytes(**kwargs)

    @patch("fastgen_profiler.mlx_guard.system_snapshot")
    def test_run_allocation_budget_rejects_oversized_shape(self, mock_snap):
        mock_snap.return_value = SystemSnapshot(
            free_bytes=10 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=10 / 128,
        )

        with pytest.raises(RuntimeMemoryAbort, match="shape budget"):
            check_run_allocation_budget(
                width=8192,
                height=8192,
                frames=64,
                guidance=7.5,
                label="oversized",
            )

    def test_prompt_budget_rejects_oversized_text_before_tokenization(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "8")

        with pytest.raises(MemoryGuardError, match="prompt text is 9 chars"):
            check_text_prompt_budget(prompt="x" * 9, label="prompt")

    def test_prompt_budget_checks_host_headroom(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "100")

        with patch("fastgen_profiler.mlx_guard.check_host_allocation_headroom") as guard:
            result = check_text_prompt_budget(
                prompt="hello",
                negative_prompt="bad",
                label="prompt",
            )

        guard.assert_called_once_with(8 * 16, label="prompt prompt text")
        assert result["prompt_chars"] == 5
        assert result["negative_prompt_chars"] == 3

    def test_prompt_budget_rejects_invalid_env(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "zero")

        with pytest.raises(MemoryGuardError, match="FASTGEN_MAX_PROMPT_CHARS"):
            check_text_prompt_budget(prompt="hello")

    def test_prompt_budget_rejects_unbounded_max_prompt_env(self, monkeypatch):
        monkeypatch.setenv("FASTGEN_MAX_PROMPT_CHARS", "65537")

        with pytest.raises(MemoryGuardError, match="FASTGEN_MAX_PROMPT_CHARS must be no greater than 65536"):
            check_text_prompt_budget(prompt="hello")

    def test_token_sequence_budget_rejects_over_model_limit(self):
        with pytest.raises(MemoryGuardError, match="token sequence is 9 tokens"):
            check_token_sequence_budget(
                token_count=9,
                max_tokens=8,
                hidden_size=4096,
                label="text",
            )

    def test_token_sequence_budget_checks_hidden_state_headroom(self):
        with patch("fastgen_profiler.mlx_guard.check_host_allocation_headroom") as guard:
            result = check_token_sequence_budget(
                token_count=4,
                max_tokens=8,
                hidden_size=16,
                label="text",
            )

        guard.assert_called_once_with(4 * 16 * 4 * 4, label="text token hidden states")
        assert result["token_count"] == 4
        assert result["max_tokens"] == 8

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"token_count": 0, "max_tokens": 8, "hidden_size": 16}, "token_count must be a positive integer"),
            ({"token_count": True, "max_tokens": 8, "hidden_size": 16}, "token_count must be a positive integer"),
            ({"token_count": 4, "max_tokens": 0, "hidden_size": 16}, "max_tokens must be a positive integer"),
            ({"token_count": 4, "max_tokens": 8.5, "hidden_size": 16}, "max_tokens must be a positive integer"),
            ({"token_count": 4, "max_tokens": 8, "hidden_size": False}, "hidden_size must be a positive integer"),
            ({"token_count": 4, "max_tokens": 8, "hidden_size": "16"}, "hidden_size must be a positive integer"),
        ],
    )
    def test_token_sequence_budget_rejects_invalid_integer_inputs(self, kwargs, message):
        with pytest.raises(MemoryGuardError, match=message):
            check_token_sequence_budget(label="text", **kwargs)  # type: ignore[arg-type]


class TestInterRunRecovery:
    @patch("fastgen_profiler.mlx_guard.time.sleep")
    @patch("fastgen_profiler.mlx_guard.check_memory_guard")
    @patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits")
    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    def test_cleanup_happens_before_limit_import(
        self,
        mock_cleanup,
        mock_limits,
        mock_check,
        _mock_sleep,
    ):
        calls: list[str] = []
        mock_check.return_value = {"free_gb": 100}
        mock_limits.side_effect = lambda label: calls.append("limits") or {"memory_limit_gb": 10}
        mock_cleanup.side_effect = lambda: calls.append("cleanup") or {"freed_gb": 0}

        inter_run_recovery(label="order")

        assert calls == ["cleanup", "limits"]

    @patch("fastgen_profiler.mlx_guard.time.sleep")
    @patch("fastgen_profiler.mlx_guard.check_memory_guard")
    @patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits")
    @patch("fastgen_profiler.mlx_guard.mlx_cleanup")
    def test_system_recovery_does_not_configure_or_import_mlx(
        self,
        mock_cleanup,
        mock_limits,
        mock_check,
        _mock_sleep,
        monkeypatch,
    ):
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx.core", None)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("system recovery must not import MLX")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        mock_check.return_value = {"free_gb": 100}
        mock_cleanup.return_value = {"freed_gb": 0, "mlx_loaded": False}

        status = inter_run_system_recovery(label="system-only")

        mock_limits.assert_not_called()
        assert status["free_gb"] == 100
        assert status["freed_gb"] == 0
        assert status["run_number"] == run_counter() + 1


class TestRunCounter:
    def test_restart_required_after_one_mlx_run_by_default(self):
        reset_run_counter()
        try:
            assert should_restart_process() is False
            assert increment_run_counter() == 1
            assert should_restart_process() is True
        finally:
            reset_run_counter()


# ---------------------------------------------------------------------------
# Adaptive batch manager (Guard 2)
# ---------------------------------------------------------------------------

class TestAdaptiveBatchManager:
    def _make_config(self, target_frames=25, target_steps=16):
        return AdaptiveBatchConfig(
            initial_frames=5,
            initial_steps=4,
            target_frames=target_frames,
            target_steps=target_steps,
        )

    def test_first_batch_is_probe(self):
        mgr = AdaptiveBatchManager(self._make_config())
        decision = mgr.next_batch(snapshot=None)
        assert decision.phase == "probe"
        assert decision.frames == 5
        assert decision.steps == 4

    def test_grows_with_good_headroom(self):
        cfg = self._make_config()
        mgr = AdaptiveBatchManager(cfg)
        mgr.next_batch(snapshot=None)  # probe

        # Good headroom: 40% free
        snap = SystemSnapshot(
            free_bytes=50 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.6,
            swap_files=0,
            free_fraction=0.4,
        )
        decision = mgr.next_batch(snapshot=snap)
        assert decision.phase == "grow"
        assert decision.frames > 5
        assert decision.steps > 4

    def test_shrinks_with_low_headroom(self):
        cfg = AdaptiveBatchConfig(
            initial_frames=20,
            initial_steps=16,
            target_frames=40,
            target_steps=32,
            min_frames=5,
            min_steps=4,
        )
        mgr = AdaptiveBatchManager(cfg)
        mgr.next_batch(snapshot=None)  # probe: 20f/16s

        # Low headroom: 10% free
        snap = SystemSnapshot(
            free_bytes=12 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.9,
            swap_files=15,
            free_fraction=0.10,
        )
        decision = mgr.next_batch(snapshot=snap)
        assert decision.phase == "shrink"
        assert decision.frames < 20  # shrunk from probe
        assert decision.steps < 16

    def test_reaches_target(self):
        cfg = self._make_config(target_frames=10, target_steps=8)
        mgr = AdaptiveBatchManager(cfg)
        mgr.next_batch(snapshot=None)  # probe: 5f/4s

        # Grow 2x: 5*2=10 frames, 4*2=8 steps → hits target
        snap = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=0.625,
        )
        decision = mgr.next_batch(snapshot=snap)
        assert decision.phase == "final"
        assert decision.frames == 10
        assert decision.steps == 8

    def test_stays_at_target(self):
        cfg = self._make_config(target_frames=10, target_steps=8)
        mgr = AdaptiveBatchManager(cfg)
        mgr.next_batch(snapshot=None)  # probe
        snap = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=0.625,
        )
        mgr.next_batch(snapshot=snap)  # reaches target

        # Next call should be steady at target
        decision = mgr.next_batch(snapshot=snap)
        assert decision.phase == "steady"
        assert decision.frames == 10
        assert decision.steps == 8

    def test_history_is_bounded_for_repeated_direct_calls(self, monkeypatch):
        import fastgen_profiler.mlx_guard as guard

        monkeypatch.setattr(guard, "MAX_ADAPTIVE_BATCH_HISTORY", 2)
        mgr = AdaptiveBatchManager(self._make_config(target_frames=10, target_steps=8))
        snap = SystemSnapshot(
            free_bytes=80 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.2,
            swap_files=0,
            free_fraction=0.625,
        )

        mgr.next_batch(snapshot=None)
        mgr.next_batch(snapshot=snap)
        mgr.next_batch(snapshot=snap)

        assert len(mgr.history) == 2
        assert [decision.phase for decision in mgr.history] == ["final", "steady"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"initial_frames": 0}, "initial_frames must be a positive integer"),
            ({"initial_steps": True}, "initial_steps must be a positive integer"),
            ({"target_frames": -1}, "target_frames must be a positive integer"),
            ({"target_steps": 4.5}, "target_steps must be a positive integer"),
            ({"min_frames": "5"}, "min_frames must be a positive integer"),
            ({"min_steps": False}, "min_steps must be a positive integer"),
            ({"headroom_grow_threshold": float("nan")}, "headroom_grow_threshold must be a finite number"),
            ({"headroom_shrink_threshold": 1.5}, "headroom_shrink_threshold must be in \\[0, 1\\]"),
            ({"max_growth_factor": 1.0}, "max_growth_factor must be a finite number greater than 1"),
            ({"initial_frames": 26}, "initial_frames cannot exceed target_frames"),
            ({"initial_steps": 17}, "initial_steps cannot exceed target_steps"),
            ({"min_frames": 26}, "min_frames cannot exceed target_frames"),
            ({"min_steps": 17}, "min_steps cannot exceed target_steps"),
        ],
    )
    def test_rejects_invalid_adaptive_config(self, kwargs, message):
        with pytest.raises(MemoryGuardError, match=message):
            AdaptiveBatchConfig(**kwargs)


class TestAdaptiveBatchConfigFromRun:
    def test_creates_config(self):
        cfg = adaptive_batch_config_from_run(target_frames=25, target_steps=16)
        assert cfg.target_frames == 25
        assert cfg.target_steps == 16
        assert cfg.initial_frames == 5
        assert cfg.initial_steps == 4

    def test_caps_initial_to_target(self):
        cfg = adaptive_batch_config_from_run(target_frames=3, target_steps=2)
        assert cfg.initial_frames == 3
        assert cfg.initial_steps == 2

    @pytest.mark.parametrize(
        ("target_frames", "target_steps", "message"),
        [
            (0, 4, "target_frames must be a positive integer"),
            (4, 0, "target_steps must be a positive integer"),
            (True, 4, "target_frames must be a positive integer"),
            (4, 1.5, "target_steps must be a positive integer"),
        ],
    )
    def test_rejects_invalid_targets(self, target_frames, target_steps, message):
        with pytest.raises(MemoryGuardError, match=message):
            adaptive_batch_config_from_run(target_frames=target_frames, target_steps=target_steps)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Headroom computation
# ---------------------------------------------------------------------------

class TestComputeHeadroom:
    def test_uses_free_fraction(self):
        snap = SystemSnapshot(
            free_bytes=30 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.5,
            swap_files=0,
            free_fraction=30 / 128,
        )
        mgr = AdaptiveBatchManager(AdaptiveBatchConfig())
        assert abs(mgr._compute_headroom(snap) - 30 / 128) < 0.01

    def test_falls_back_to_pressure(self):
        snap = SystemSnapshot(
            free_bytes=None,
            total_bytes=None,
            pressure=0.3,
            swap_files=0,
            free_fraction=None,
        )
        mgr = AdaptiveBatchManager(AdaptiveBatchConfig())
        assert abs(mgr._compute_headroom(snap) - 0.7) < 0.01

    def test_returns_none_for_no_data(self):
        snap = SystemSnapshot(
            free_bytes=None,
            total_bytes=None,
            pressure=None,
            swap_files=None,
            free_fraction=None,
        )
        mgr = AdaptiveBatchManager(AdaptiveBatchConfig())
        assert mgr._compute_headroom(snap) is None


# ---------------------------------------------------------------------------
# CLI adaptive spec adjustment
# ---------------------------------------------------------------------------

from fastgen_profiler.cli import (
    PresetRun,
    ProfileRunSpec,
    _adaptive_adjust_spec,
)


def _make_spec(frames=16, steps=8, label="test-spec"):
    return ProfileRunSpec(
        preset="smoke",
        variant_label=label,
        run=PresetRun(
            width=512,
            height=512,
            frames=frames,
            steps=steps,
            guidance=3.5,
            quant="none",
            cache="none",
            compile="off",
            save_video=False,
        ),
    )


class TestAdaptiveAdjustSpec:
    def test_stub_backend_never_adjusted(self):
        spec = _make_spec(frames=81, steps=40)
        state = {"last_snapshot": SystemSnapshot(
            free_bytes=1 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.95,
            swap_files=20,
            free_fraction=1 / 128,
        ), "shrunk_specs": set()}
        result = _adaptive_adjust_spec(spec, state, backend_name="stub")
        assert result.run.frames == 81
        assert result.run.steps == 40

    def test_no_snapshot_returns_original(self):
        spec = _make_spec()
        state = {"last_snapshot": None, "shrunk_specs": set()}
        result = _adaptive_adjust_spec(spec, state, backend_name="mlx")
        assert result is spec

    def test_low_headroom_halves_frames_steps(self):
        spec = _make_spec(frames=16, steps=8)
        state = {"last_snapshot": SystemSnapshot(
            free_bytes=5 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.92,
            swap_files=10,
            free_fraction=5 / 128,
        ), "shrunk_specs": set()}
        result = _adaptive_adjust_spec(spec, state, backend_name="mlx")
        assert result.run.frames == 8
        assert result.run.steps == 4

    def test_good_headroom_restores_shrunk(self):
        spec = _make_spec(frames=16, steps=8)
        state = {"last_snapshot": SystemSnapshot(
            free_bytes=60 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.4,
            swap_files=0,
            free_fraction=60 / 128,
        ), "shrunk_specs": {"test-spec"}}
        result = _adaptive_adjust_spec(spec, state, backend_name="mlx")
        assert result.run.frames == 16
        assert result.run.steps == 8

    def test_moderate_pressure_reduces_25pct(self):
        spec = _make_spec(frames=24, steps=16)
        # 20% headroom → moderate (between 15% and 25%)
        state = {"last_snapshot": SystemSnapshot(
            free_bytes=25 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.8,
            swap_files=5,
            free_fraction=25 / 128,
        ), "shrunk_specs": set()}
        result = _adaptive_adjust_spec(spec, state, backend_name="mlx")
        assert result.run.frames == 18  # 24 * 0.75
        assert result.run.steps == 12   # 16 * 0.75

    def test_minimum_frames_steps_enforced(self):
        spec = _make_spec(frames=5, steps=3)
        state = {"last_snapshot": SystemSnapshot(
            free_bytes=1 * 1024 ** 3,
            total_bytes=128 * 1024 ** 3,
            pressure=0.95,
            swap_files=20,
            free_fraction=1 / 128,
        ), "shrunk_specs": set()}
        result = _adaptive_adjust_spec(spec, state, backend_name="mlx")
        assert result.run.frames >= 4
        assert result.run.steps >= 2


class TestAdapterRuntimeGuards:
    def test_runtime_modules_do_not_top_level_import_mlx_or_mlx_video(self):
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        checked_roots = [repo_root / "src" / "fastgen_profiler", repo_root / "scripts"]
        blocked = {"mlx", "mlx_video", "mlx_lm"}
        offenders: list[str] = []

        for root in checked_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.py")):
                module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in module.body:
                    imported: list[str] = []
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        imported = [node.module]
                    for name in imported:
                        package = name.split(".", 1)[0]
                        if package in blocked:
                            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}:{name}")

        assert offenders == []

    def test_adapter_module_imports_do_not_import_numpy(self):
        import os
        import subprocess
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        code = """
import builtins
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise AssertionError("adapter module import must not import numpy")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import fastgen_profiler.backends.ltx23_mlx_adapter
import fastgen_profiler.backends.wan22_mlx_adapter
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert result.returncode == 0, result.stderr

    def test_video_frame_normalization_uses_in_place_numpy_ops(self):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import _normalize_video_frames as ltx_normalize
        from fastgen_profiler.backends.wan22_mlx_adapter import _normalize_video_frames as wan_normalize

        class TrackingNumpy:
            uint8 = np.uint8

            def __init__(self):
                self.out_targets: list[object] = []

            def add(self, value, amount, *, out):
                self.out_targets.append(out)
                return np.add(value, amount, out=out)

            def multiply(self, value, amount, *, out):
                self.out_targets.append(out)
                return np.multiply(value, amount, out=out)

            def clip(self, value, min_value, max_value, *, out):
                self.out_targets.append(out)
                return np.clip(value, min_value, max_value, out=out)

        for normalize in (ltx_normalize, wan_normalize):
            tracker = TrackingNumpy()
            frames = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
            result = normalize(tracker, frames)

            assert tracker.out_targets == [frames, frames, frames]
            assert result.dtype == np.uint8
            assert result.tolist() == [[0, 127, 255]]

    def test_base_synchronize_does_not_import_mlx(self, monkeypatch):
        from fastgen_profiler.backends.base import synchronize_mlx

        sys.modules.pop("mlx.core", None)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("synchronize must not initialize MLX")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        synchronize_mlx(object())

    def test_base_mlx_memory_snapshot_sanitizes_invalid_counters(self, monkeypatch):
        from fastgen_profiler.backends.base import mlx_memory_snapshot

        fake_mx = types.SimpleNamespace(
            get_active_memory=lambda: -1,
            get_peak_memory=lambda: "100",
            get_cache_memory=lambda: 50,
        )
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        assert mlx_memory_snapshot() == {
            "active_memory": None,
            "peak_memory": None,
            "cache_memory": 50,
        }

    def test_adapter_synchronize_does_not_import_mlx(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        sys.modules.pop("mlx.core", None)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("adapter synchronize must not initialize MLX")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        ).synchronize(None)
        LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        ).synchronize(object())
        Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        ).synchronize(None)
        Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        ).synchronize(object())

    def test_adapter_synchronize_uses_loaded_mlx_without_reconfiguring_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        calls: list[object] = []
        fake_mx = types.SimpleNamespace(eval=lambda target: calls.append(target))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            ltx_target = object()
            wan_target = object()
            LTX23MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            ).synchronize(ltx_target)
            Wan22MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            ).synchronize(wan_target)

        assert calls == [ltx_target, wan_target]
        limits.assert_not_called()

    def test_base_synchronize_aborts_and_cleans_up_when_loaded_mlx_eval_fails(self, monkeypatch):
        from fastgen_profiler.backends.base import synchronize_mlx

        cleanup_calls: list[str] = []
        fake_mx = types.SimpleNamespace(
            eval=lambda target: (_ for _ in ()).throw(RuntimeError("metal sync failed")),
            clear_cache=lambda: cleanup_calls.append("clear"),
        )
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with pytest.raises(RuntimeMemoryAbort, match="MLX synchronization failed"):
            synchronize_mlx(object())

        assert cleanup_calls == ["clear"]

    def test_adapter_synchronize_aborts_and_cleans_up_when_loaded_mlx_eval_fails(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        cleanup_calls: list[str] = []
        fake_mx = types.SimpleNamespace(
            eval=lambda target: (_ for _ in ()).throw(RuntimeError("metal sync failed")),
            clear_cache=lambda: cleanup_calls.append("clear"),
        )
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        with pytest.raises(RuntimeMemoryAbort, match="ltx2.3 synchronize"):
            LTX23MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            ).synchronize(object())
        with pytest.raises(RuntimeMemoryAbort, match="wan2.2 synchronize"):
            Wan22MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            ).synchronize(object())

        assert cleanup_calls == ["clear", "clear"]

    def test_adapter_phase_eval_failure_aborts_and_cleans_up(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        cleanup_calls: list[str] = []
        fake_mx = types.SimpleNamespace(
            eval=lambda *args: (_ for _ in ()).throw(RuntimeError("metal eval failed")),
            clear_cache=lambda: cleanup_calls.append("clear"),
            random=types.SimpleNamespace(seed=lambda seed: None, normal=lambda shape: object()),
        )
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        ltx.model = object()
        ltx.config = types.SimpleNamespace(in_channels=128)
        ltx._mlx_runtime_ready = True
        ltx._check_memory = lambda phase: None  # type: ignore[method-assign]
        ltx._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match="ltx2.3 latent_init"):
            ltx.init_latents(seed=1, width=256, height=256, frames=4)

        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan.mx = fake_mx
        wan.latent_shape = (16, 1, 1, 1)
        wan._check_memory = lambda phase: None  # type: ignore[method-assign]
        wan._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match="wan2.2 latent_init"):
            wan.init_latents(seed=1, width=256, height=256, frames=4)

        assert cleanup_calls == ["clear", "clear"]

    def test_mlx_backend_watchdog_fails_closed_when_guard_unavailable(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.mlx import MLXBackend
        from fastgen_profiler.metrics import RunConfig

        config = RunConfig(
            model="wan2.2",
            backend="mlx",
            model_path=str(tmp_path),
            model_id=None,
            model_source_root=None,
            prompt="prompt",
            negative_prompt="",
            seed=1,
            width=512,
            height=512,
            frames=9,
            fps=24,
            steps=1,
            guidance=1.0,
            quant="none",
            cache="none",
            compile="off",
            output_dir=tmp_path / "outputs",
            result_jsonl=tmp_path / "results.jsonl",
            save_video=False,
            dry_run=False,
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "fastgen_profiler.mlx_guard":
                raise ImportError("guard unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeError, match="mlx_guard unavailable before MLX runtime watchdog"):
            MLXBackend().run(config, run_id="run", timestamp_utc="2026-06-02T00:00:00Z", machine={})

    def test_wan22_load_model_requires_runtime_guard_before_work(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        (tmp_path / "tokenizer").mkdir()
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        calls: list[str] = []
        pipe._check_file_load = lambda path, phase: calls.append(phase)  # type: ignore[method-assign]
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        try:
            with (
                patch(
                    "fastgen_profiler.mlx_guard.check_memory_guard",
                    side_effect=lambda label: calls.append("system") or {"free_gb": 100},
                ),
                patch(
                    "fastgen_profiler.mlx_guard.check_run_allocation_budget",
                    side_effect=lambda **kwargs: calls.append("budget") or {"shape_floor_gb": 1},
                ),
                patch(
                    "fastgen_profiler.mlx_guard.configure_mlx_resource_limits",
                    side_effect=lambda label: calls.append("limits") or (_ for _ in ()).throw(MemoryGuardError("guard first")),
                ),
            ):
                with pytest.raises(MemoryGuardError, match="guard first"):
                    pipe.load_model()
        finally:
            monkeypatch.undo()

        assert calls == [
            "preflight t5_encoder",
            "preflight model",
            "preflight config",
            "budget",
            "system",
            "budget",
            "limits",
        ]

    def test_wan22_load_model_budgets_aligned_config_shape_after_limits_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=1,
            height=1,
            frames=4,
            steps=1,
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "dual_model": False,
                    "patch_size": [1, 1024, 1024],
                    "vae_stride": [1, 8, 8],
                    "max_area": 0,
                    "vae_z_dim": 48,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        config = types.SimpleNamespace(
            dual_model=False,
            patch_size=(1, 1024, 1024),
            vae_stride=(1, 8, 8),
            max_area=0,
            vae_z_dim=48,
        )
        budget_calls: list[dict[str, object]] = []
        calls: list[str] = []
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        def load_config(model_path):
            calls.append("load_config")
            assert "limits" in calls
            return config, None

        monkeypatch.setattr("fastgen_profiler.backends.wan22_mlx_adapter._load_config", load_config)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video") and "limits" not in calls:
                raise AssertionError("Wan config preflight must not import mlx_video before limits")
            if name in {"mlx", "mlx.core"}:
                calls.append("mlx_import")
                raise RuntimeError("stop before mlx import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with (
            patch("fastgen_profiler.mlx_guard.check_memory_guard", return_value={"free_gb": 100}),
            patch(
                "fastgen_profiler.mlx_guard.check_run_allocation_budget",
                side_effect=lambda **kwargs: budget_calls.append(kwargs) or {"shape_floor_gb": 1},
            ),
            patch(
                "fastgen_profiler.mlx_guard.configure_mlx_resource_limits",
                side_effect=lambda label: calls.append("limits") or {"memory_limit_gb": 1},
            ),
        ):
            with pytest.raises(RuntimeError, match="stop before mlx import"):
                pipe.load_model()

        assert calls == ["limits", "load_config", "mlx_import"]
        assert len(budget_calls) == 3
        assert budget_calls[0]["width"] == 8192
        assert budget_calls[0]["height"] == 8192
        assert budget_calls[1]["width"] == 8192
        assert budget_calls[1]["height"] == 8192
        assert budget_calls[2]["width"] == 8192
        assert budget_calls[2]["height"] == 8192
        assert budget_calls[2]["frames"] == 4

    def test_ltx23_load_model_requires_runtime_guard_before_work(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        calls: list[str] = []
        pipe._check_directory_load = lambda path, phase: calls.append(phase)  # type: ignore[method-assign]
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        try:
            with (
                patch(
                    "fastgen_profiler.mlx_guard.check_memory_guard",
                    side_effect=lambda label: calls.append("system") or {"free_gb": 100},
                ),
                patch(
                    "fastgen_profiler.mlx_guard.check_run_allocation_budget",
                    side_effect=lambda **kwargs: calls.append("budget") or {"shape_floor_gb": 1},
                ),
                patch(
                    "fastgen_profiler.mlx_guard.configure_mlx_resource_limits",
                    side_effect=lambda label: calls.append("limits") or (_ for _ in ()).throw(MemoryGuardError("guard first")),
                ),
            ):
                with pytest.raises(MemoryGuardError, match="guard first"):
                    pipe.load_model()
        finally:
            monkeypatch.undo()

        assert calls == ["preflight transformer", "system", "budget", "limits"]

    def test_ltx23_load_model_preflights_config_latent_shape_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(
            json.dumps({"in_channels": 60_000}),
            encoding="utf-8",
        )
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (  # type: ignore[method-assign]
            None
            if phase == "preflight transformer config"
            else (_ for _ in ()).throw(RuntimeMemoryAbort(f"{phase} too large"))
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="config latent tensor too large"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_transformer_config_file_size_preflight_runs_before_json_read(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text("{not-json}", encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort(f"{phase} too large")
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="preflight transformer config too large"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_transformer_config_rejects_oversized_json_before_read(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(
            " " * (ltx23_mlx_adapter._MAX_CONFIG_JSON_BYTES + 1),
            encoding="utf-8",
        )
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="above safe config limit"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_transformer_config_rejects_too_many_json_items_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import _read_bounded_json_config

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {f"key_{index}": index for index in range(ltx23_mlx_adapter._MAX_CONFIG_JSON_ITEMS + 1)}
            ),
            encoding="utf-8",
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="safe item limit"):
                _read_bounded_json_config(config_path, "preflight transformer config")

        limits.assert_not_called()

    def test_ltx23_transformer_config_rejects_deep_json_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import _read_bounded_json_config

        value: object = 0
        for _ in range(ltx23_mlx_adapter._MAX_CONFIG_JSON_DEPTH + 1):
            value = {"nested": value}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(value), encoding="utf-8")

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="safe depth"):
                _read_bounded_json_config(config_path, "preflight transformer config")

        limits.assert_not_called()

    def test_ltx23_load_model_rejects_unsafe_structural_config_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(
            json.dumps({"hidden_size": 1_000_000_000}),
            encoding="utf-8",
        )
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="hidden_size=1000000000 exceeds safe structural dimension"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_load_model_rejects_non_positive_structural_config_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"hidden_size": 0}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="hidden_size=0 must be a positive structural dimension"):
                pipe.load_model()

        limits.assert_not_called()

    @pytest.mark.parametrize("value", [1.5, "128"])
    def test_ltx23_structural_config_rejects_non_integer_values(self, value):
        from fastgen_profiler.backends.ltx23_mlx_adapter import _positive_structural_int

        with pytest.raises(RuntimeMemoryAbort, match="hidden_size=.*must be a positive structural dimension"):
            _positive_structural_int(value, "hidden_size")

    @pytest.mark.parametrize(
        ("helper_name", "value", "message"),
        [
            ("_positive_int", 1.5, "dim=.*must be a positive integer"),
            ("_positive_int", "4096", "dim=.*must be a positive integer"),
            ("_non_negative_int", 1.5, "max_area=.*must be zero or a positive integer"),
            ("_non_negative_int", "0", "max_area=.*must be zero or a positive integer"),
        ],
    )
    def test_wan22_structural_config_rejects_non_integer_values(self, helper_name, value, message):
        import fastgen_profiler.backends.wan22_mlx_adapter as wan_adapter

        helper = getattr(wan_adapter, helper_name)

        with pytest.raises(RuntimeMemoryAbort, match=message):
            helper(value, "dim" if helper_name == "_positive_int" else "max_area")

    def test_ltx23_latent_shape_uses_ceil_grid_for_unaligned_shape(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=257,
            height=257,
            frames=4,
            steps=1,
        )
        pipe.config = types.SimpleNamespace(in_channels=128)

        assert pipe._expected_latent_shape() == (1, 128, 4, 33, 33)

    def test_ltx23_load_model_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")

        class FakeConfig:
            model_type = "ltx2.3"

            def __init__(self, **kwargs):
                self.in_channels = kwargs.get("in_channels", 128)

        class FakeModel:
            def __init__(self, config):
                self.config = config

            def parameters(self):
                return {"weight": object()}

            def load_weights(self, items, strict=False):
                raise RuntimeError("ltx load failed after runtime opened")

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        config_module = types.ModuleType("mlx_video.models.ltx_2.config")
        config_module.LTXModelConfig = FakeConfig
        model_module = types.ModuleType("mlx_video.models.ltx_2.ltx_2")
        model_module.LTXModel = FakeModel
        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {"weight": object()}
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
        monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="ltx load failed"):
            pipe.load_model()

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_load_model_import_failure_after_runtime_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise RuntimeError("ltx mlx import failed after runtime opened")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="ltx mlx import failed"):
            pipe.load_model()

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_load_model_rejects_unmatched_transformer_weights(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")

        class FakeConfig:
            model_type = "ltx2.3"

            def __init__(self, **kwargs):
                self.in_channels = kwargs.get("in_channels", 128)

        class FakeModel:
            def __init__(self, config):
                self.config = config

            def parameters(self):
                return {"expected.weight": object()}

            def load_weights(self, items, strict=False):
                raise AssertionError("load_weights must not run without matched transformer weights")

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        config_module = types.ModuleType("mlx_video.models.ltx_2.config")
        config_module.LTXModelConfig = FakeConfig
        model_module = types.ModuleType("mlx_video.models.ltx_2.ltx_2")
        model_module.LTXModel = FakeModel
        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {"unmatched.weight": object()}
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
        monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="transformer weights did not match"):
            pipe.load_model()

    def test_ltx23_load_model_streams_transformer_weight_filter(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        loaded_weights = []

        class GuardedItems:
            def __iter__(self):
                yield ("expected.weight", object())
                yield ("expected.bias", object())
                yield ("expected.input_scale", object())
                yield ("unmatched.weight", object())

            def __len__(self):
                raise AssertionError("transformer weights must not be list-materialized before load_weights")

        class FakeWeights:
            def items(self):
                return GuardedItems()

        class FakeConfig:
            model_type = "ltx2.3"

            def __init__(self, **kwargs):
                self.in_channels = kwargs.get("in_channels", 128)

        class FakeModel:
            def __init__(self, config):
                self.config = config

            def parameters(self):
                return {"expected.weight": object(), "expected.bias": object()}

            def load_weights(self, items, strict=False):
                assert strict is False
                assert not isinstance(items, list)
                loaded_weights.append(list(items))

            def eval(self):
                return None

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        config_module = types.ModuleType("mlx_video.models.ltx_2.config")
        config_module.LTXModelConfig = FakeConfig
        model_module = types.ModuleType("mlx_video.models.ltx_2.ltx_2")
        model_module.LTXModel = FakeModel
        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: FakeWeights()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
        monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        assert pipe.load_model()["model_type"] == "ltx2.3"
        assert [[key for key, _value in loaded] for loaded in loaded_weights] == [
            ["expected.weight", "expected.bias"]
        ]

    def test_ltx23_load_model_preflights_model_config_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(
            json.dumps(
                {
                    "in_channels": 128,
                    "hidden_size": 60_000,
                    "intermediate_size": 60_000,
                    "num_layers": 1,
                    "num_attention_heads": 1,
                }
            ),
            encoding="utf-8",
        )
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (
            (_ for _ in ()).throw(RuntimeMemoryAbort("model config too large"))
            if phase == "config model tensor"
            else None
        )  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="model config too large"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_missing_transformer_config_blocks_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(FileNotFoundError, match="transformer config not found"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_missing_transformer_weights_blocks_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(FileNotFoundError, match="transformer weights not found"):
                pipe.load_model()

        limits.assert_not_called()

    def test_wan22_load_model_preflights_model_config_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "patch_size": [1, 2, 2],
                    "vae_stride": [4, 16, 16],
                    "max_area": 0,
                    "vae_z_dim": 48,
                    "dim": 60_000,
                    "ffn_dim": 60_000,
                    "num_layers": 1,
                    "num_heads": 1,
                    "text_dim": 60_000,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (
            (_ for _ in ()).throw(RuntimeMemoryAbort("wan model config too large"))
            if phase == "config model tensor"
            else None
        )  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="wan model config too large"):
                pipe.load_model()

        limits.assert_not_called()

    def test_adapter_runtime_helpers_fail_closed_when_guard_import_fails(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "fastgen_profiler.mlx_guard":
                raise ImportError("guard unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        for pipe, label in ((ltx, "ltx2.3"), (wan, "wan2.2")):
            with pytest.raises(RuntimeError, match=rf"memory guard unavailable before {label} denoise"):
                pipe._check_memory("denoise")
            with pytest.raises(RuntimeError, match=rf"memory guard unavailable before {label} load weights"):
                pipe._check_host_allocation(1, "load weights")

    def test_adapter_prepare_prompt_checks_text_budget(self, tmp_path, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        calls: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            mlx_guard,
            "check_text_prompt_budget",
            lambda *, prompt, negative_prompt, label: calls.append((prompt, negative_prompt, label)),
        )

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        ltx.config = types.SimpleNamespace()
        wan.config = types.SimpleNamespace(sample_neg_prompt="default negative")

        assert ltx.prepare_prompt(prompt="p", negative_prompt=None) == {"prompt": "p", "negative_prompt": ""}
        assert wan.prepare_prompt(prompt="w", negative_prompt=None) == {
            "prompt": "w",
            "negative_prompt": "default negative",
        }
        assert calls == [
            ("p", "", "ltx2.3 prompt"),
            ("w", "default negative", "wan2.2 prompt"),
        ]

    def test_direct_encode_text_checks_raw_prompt_budget_before_tokenizer(self, tmp_path, monkeypatch):
        import fastgen_profiler.mlx_guard as mlx_guard
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        ltx.model = object()
        ltx.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16, in_channels=128)
        wan.mx = object()
        wan.config = types.SimpleNamespace(text_len=128, dim=4096)
        wan.model = object()
        wan.t5_encoder = object()
        wan.tokenizer = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tokenizer must be rejected first")
        )
        wan.seq_len = 1

        monkeypatch.setattr(
            mlx_guard,
            "check_text_prompt_budget",
            lambda **kwargs: (_ for _ in ()).throw(MemoryGuardError("prompt too large")),
        )

        with pytest.raises(MemoryGuardError, match="prompt too large"):
            ltx.encode_text({"prompt": "x" * 100_000, "negative_prompt": ""})
        with pytest.raises(MemoryGuardError, match="prompt too large"):
            wan.encode_text({"prompt": "x" * 100_000, "negative_prompt": ""})

    def test_ltx23_latent_init_preflights_tensor_allocation(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        fake_mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(AssertionError("latent allocation must be preflighted")),
            ),
            eval=lambda *args: None,
            get_active_memory=lambda: 0,
            get_cache_memory=lambda: 0,
        )
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_host_allocation = lambda required_bytes, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("latent too large")
        )

        with pytest.raises(RuntimeMemoryAbort, match="latent too large"):
            pipe.init_latents(seed=1, width=256, height=256, frames=4)

    def test_ltx23_latent_init_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        fake_mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(RuntimeError("ltx latent init failed")),
            ),
            eval=lambda *args: None,
            get_active_memory=lambda: 0,
            get_cache_memory=lambda: 0,
        )
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="ltx latent init failed"):
            pipe.init_latents(seed=1, width=256, height=256, frames=4)

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_latent_init_rejects_shape_args_that_bypass_pipeline_budget(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        fake_mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(AssertionError("latent allocation must be rejected first")),
            ),
            eval=lambda *args: None,
        )
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True

        with pytest.raises(RuntimeError, match="latent_init shape .* pipeline shape"):
            pipe.init_latents(seed=1, width=8192, height=8192, frames=64)

    def test_ltx23_latent_init_requires_loaded_model(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("latent init must not import MLX before load_model")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeError, match="init_latents called before load_model"):
            pipe.init_latents(seed=1, width=256, height=256, frames=4)

    def test_ltx23_denoise_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class FakeLatents:
            dtype = "float32"
            shape = (1, 128, 4, 32, 32)

        fake_mx = types.SimpleNamespace(
            float32="float32",
            array=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ltx denoise failed")),
            get_active_memory=lambda: 0,
            get_cache_memory=lambda: 0,
        )
        transformer_module = types.ModuleType("mlx_video.models.ltx_2.transformer")
        transformer_module.Modality = lambda **kwargs: object()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, transformer_module.__name__, transformer_module)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe.context_emb = object()
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="ltx denoise failed"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_runtime_phases_require_loaded_config_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("runtime phase must not import MLX before load_model config exists")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        class FakeLatents:
            dtype = "float32"
            shape = (1, 128, 4, 32, 32)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()

        with pytest.raises(RuntimeError, match="encode_text called before load_model"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})
        with pytest.raises(RuntimeError, match="denoise_step called before load_model"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")
        with pytest.raises(RuntimeError, match="decode called before load_model"):
            pipe.decode(FakeLatents())

    def test_ltx23_denoise_preflights_intermediate_tensors_before_mx_array(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class FakeLatents:
            dtype = "float32"
            shape = (1, 128, 4, 32, 32)

        fake_mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mx.array must be preflighted")),
            float32="float32",
        )
        transformer_module = types.ModuleType("mlx_video.models.ltx_2.transformer")
        transformer_module.Modality = object
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx_video.models.ltx_2.transformer", transformer_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("denoise tensors too large")
        )

        with pytest.raises(RuntimeMemoryAbort, match="denoise tensors too large"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

    def test_ltx23_denoise_preflight_reuses_validated_latent_shape(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class SinglePassShape:
            def __init__(self, dims):
                self._dims = dims
                self._iterations = 0

            def __iter__(self):
                self._iterations += 1
                if self._iterations > 1:
                    raise AssertionError("denoise must not re-read latent shape after validation")
                yield from self._dims

        class FakeLatents:
            dtype = "float32"
            shape = SinglePassShape((1, 128, 4, 32, 32))

        fake_mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mx.array must be preflighted")),
            float32="float32",
        )
        transformer_module = types.ModuleType("mlx_video.models.ltx_2.transformer")
        transformer_module.Modality = object
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx_video.models.ltx_2.transformer", transformer_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("denoise tensors too large")
        )

        with pytest.raises(RuntimeMemoryAbort, match="denoise tensors too large"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

    def test_ltx23_denoise_rejects_latent_shape_mismatch_before_mx_array(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class FakeLatents:
            dtype = "float32"
            shape = (1, 128, 64, 1024, 1024)

        fake_mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mx.array must be rejected first")),
            float32="float32",
        )
        transformer_module = types.ModuleType("mlx_video.models.ltx_2.transformer")
        transformer_module.Modality = object
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx_video.models.ltx_2.transformer", transformer_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe.model = object()
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="latent shape .* expected"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

    def test_ltx23_encode_text_resolves_text_assets_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path / "model",
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)

        def missing_text_encoder(dest, auto_download):
            raise FileNotFoundError("text encoder missing")

        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_text_encoder_download.ensure_text_encoder",
            missing_text_encoder,
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("text asset preflight must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(FileNotFoundError, match="text encoder missing"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_preflights_text_projection_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x" * 1024)
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)
        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)
        pipe._check_file_load = lambda path, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("text projection too large")
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("text projection preflight must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="text projection too large"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_preflights_text_projection_construct_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x")
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)
        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=60_000, cross_attention_dim=60_000)
        pipe._check_host_allocation = lambda required_bytes, phase: (  # type: ignore[method-assign]
            (_ for _ in ()).throw(RuntimeMemoryAbort("text projection construct too large"))
            if phase == "text_projection construct"
            else None
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("text projection construct preflight must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="text projection construct too large"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_missing_text_projection_blocks_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)
        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("missing text projection must be rejected before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(FileNotFoundError, match="text projection weights not found"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_rejects_token_sequence_before_text_projection_import(self, tmp_path, monkeypatch):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "vocab_size": 32,
                        "num_attention_heads": 1,
                        "max_position_embeddings": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": np.array([[1, 2, 3]])}

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path, local_files_only):
                return FakeTokenizer()

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("token budget must run before text projection MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "transformers", transformers_module)
        monkeypatch.setattr(builtins, "__import__", guarded_import)

        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]

        with pytest.raises(MemoryGuardError, match="token sequence is 3 tokens"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_checks_runtime_before_text_projection_safetensors_load(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x")
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)

        class FakeTextProjection:
            def __init__(self, *args):
                pass

            def parameters(self):
                return {}

            def eval(self):
                return None

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: (_ for _ in ()).throw(
            AssertionError("load_safetensors must be rejected first")
        )
        projection_module = types.ModuleType("mlx_video.models.ltx_2.text_projection")
        projection_module.PixArtAlphaTextProjection = FakeTextProjection
        for name in ["mlx_video", "mlx_video.models", "mlx_video.models.ltx_2"]:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, projection_module.__name__, projection_module)

        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: (  # type: ignore[method-assign]
            (_ for _ in ()).throw(RuntimeMemoryAbort("runtime before safetensors"))
            if phase == "text_projection load_safetensors before"
            else None
        )

        with pytest.raises(RuntimeMemoryAbort, match="runtime before safetensors"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_rejects_unmatched_text_projection_weights(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x")
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)

        class FakeTextProjection:
            def __init__(self, *args):
                pass

            def parameters(self):
                return {"expected.weight": object()}

            def eval(self):
                raise AssertionError("text projection must not eval with unmatched weights")

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {"unmatched.weight": object()}
        projection_module = types.ModuleType("mlx_video.models.ltx_2.text_projection")
        projection_module.PixArtAlphaTextProjection = FakeTextProjection
        for name in ["mlx_video", "mlx_video.models", "mlx_video.models.ltx_2"]:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, projection_module.__name__, projection_module)

        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16)
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="text projection weights did not match"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_ltx23_encode_text_streams_text_projection_weight_filter(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x")
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        class GuardedItems:
            def __init__(self):
                self.materialized = False
                self.iterated = False

            def __iter__(self):
                if self.iterated:
                    raise AssertionError("text projection weight items must be consumed in a single pass")
                self.iterated = True
                if self.materialized:
                    raise AssertionError("text projection weights must not be list-materialized before load_weights")
                yield ("expected.weight", object())
                yield ("unmatched.weight", object())

            def __len__(self):
                self.materialized = True
                raise AssertionError("text projection weights must not be list-materialized before load_weights")

        guarded_items = GuardedItems()
        loaded_weights = []

        class FakeWeights:
            def items(self):
                return guarded_items

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)

        class FakeTextProjection:
            def __init__(self, *args):
                pass

            def parameters(self):
                return {"expected.weight": object()}

            def load_weights(self, items):
                assert not isinstance(items, list)
                loaded_weights.append(list(items))

            def eval(self):
                return None

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: FakeWeights()
        projection_module = types.ModuleType("mlx_video.models.ltx_2.text_projection")
        projection_module.PixArtAlphaTextProjection = FakeTextProjection
        for name in ["mlx_video", "mlx_video.models", "mlx_video.models.ltx_2"]:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, projection_module.__name__, projection_module)

        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16, in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._encode_with_gemma3 = lambda prompt, text_encoder_dir, tokenizer_dir, in_features: object()  # type: ignore[method-assign]

        assert pipe.encode_text({"prompt": "prompt", "negative_prompt": ""}) is pipe.context_emb
        assert len(loaded_weights) == 1
        assert [key for key, _value in loaded_weights[0]] == ["expected.weight"]

    def test_ltx23_filtered_weight_items_preserves_single_pass_iterators(self):
        from fastgen_profiler.backends.ltx23_mlx_adapter import _filtered_weight_items

        def weight_items():
            yield ("unmatched.weight", object())
            yield ("expected.weight", "first")
            yield ("expected.bias", "second")

        filtered = _filtered_weight_items(
            weight_items(),
            allowed_names={"expected.weight", "expected.bias"},
            label="test weights",
        )

        assert filtered.match_count == 1
        assert list(filtered) == [("expected.weight", "first"), ("expected.bias", "second")]

    def test_ltx23_parameter_name_flatten_fails_closed_when_unbounded(self, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PARAMETER_NAMES", 2)

        with pytest.raises(RuntimeMemoryAbort, match="more than 2 parameter names"):
            ltx23_mlx_adapter._flatten_parameter_names(
                {
                    "layer0": {"weight": object()},
                    "layer1": {"weight": object()},
                    "layer2": {"weight": object()},
                },
                label="test model parameters",
            )

    def test_ltx23_parameter_name_add_fails_closed_when_unbounded(self, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PARAMETER_NAMES", 1)
        names: set[str] = set()

        ltx23_mlx_adapter._add_parameter_name(names, "first.weight", label="test projection parameters")

        with pytest.raises(RuntimeMemoryAbort, match="test projection parameters"):
            ltx23_mlx_adapter._add_parameter_name(
                names,
                "second.weight",
                label="test projection parameters",
            )

    def test_ltx23_parameter_name_add_rejects_long_names(self, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PARAMETER_NAME_CHARS", 4)

        with pytest.raises(RuntimeMemoryAbort, match="name exceeds 4"):
            ltx23_mlx_adapter._add_parameter_name(
                set(),
                "too-long",
                label="test projection parameters",
            )

    def test_ltx23_parameter_name_flatten_rejects_deep_nesting(self, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PARAMETER_NAME_DEPTH", 2)
        params: object = object()
        for index in range(4):
            params = {f"layer{index}": params}

        with pytest.raises(RuntimeMemoryAbort, match="nesting exceeds 2"):
            ltx23_mlx_adapter._flatten_parameter_names(params, label="test model parameters")

    def test_ltx23_parameter_name_flatten_rejects_unsafe_keys_without_repr_or_str(self):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        class UnsafeKey:
            def __repr__(self):
                raise AssertionError("parameter-name guard must not call repr on unknown keys")

            def __str__(self):
                raise AssertionError("parameter-name guard must not call str on unknown keys")

        with pytest.raises(RuntimeMemoryAbort, match="UnsafeKey"):
            ltx23_mlx_adapter._flatten_parameter_names(
                {UnsafeKey(): object()},
                label="test model parameters",
            )

    def test_ltx23_weight_filter_rejects_unsafe_keys_without_repr_or_str(self):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        class UnsafeKey:
            def __repr__(self):
                raise AssertionError("weight-key guard must not call repr on unknown keys")

            def __str__(self):
                raise AssertionError("weight-key guard must not call str on unknown keys")

        with pytest.raises(RuntimeMemoryAbort, match="UnsafeKey"):
            list(
                ltx23_mlx_adapter._filtered_weight_items(
                    [(UnsafeKey(), object())],
                    allowed_names={"expected.weight"},
                    label="test weights",
                )
            )

    def test_ltx23_weight_filter_rejects_malformed_items_without_repr(self):
        from fastgen_profiler.backends import ltx23_mlx_adapter

        class UnsafeItem:
            def __iter__(self):
                raise TypeError("not a pair")

            def __repr__(self):
                raise AssertionError("malformed weight item repr must not be called")

        with pytest.raises(RuntimeMemoryAbort, match="malformed weight item"):
            list(
                ltx23_mlx_adapter._filtered_weight_items(
                    [UnsafeItem()],
                    allowed_names={"expected.weight"},
                    label="test weights",
                )
            )

    def test_ltx23_encode_text_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_path = tmp_path / "model"
        text_proj_dir = model_path / "text_projections"
        text_proj_dir.mkdir(parents=True)
        (text_proj_dir / "model.safetensors").write_bytes(b"x")
        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        class FakeProjection:
            def __init__(self, in_features, hidden_size):
                pass

            def parameters(self):
                return {"expected.weight": object()}

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: (_ for _ in ()).throw(
            RuntimeError("text projection load failed after runtime opened")
        )
        projection_module = types.ModuleType("mlx_video.models.ltx_2.text_projection")
        projection_module.PixArtAlphaTextProjection = FakeProjection
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, projection_module.__name__, projection_module)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=model_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16, in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="text projection load failed"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_text_encoder_rejects_unmatched_weights_before_encoding(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        fake_mx = types.SimpleNamespace(
            array=lambda value: (_ for _ in ()).throw(AssertionError("mx.array must not run with unmatched weights")),
            clear_cache=lambda: None,
            eval=lambda *args: None,
        )

        class FakeModelArgs:
            def __init__(self, **kwargs):
                pass

        class FakeGemma3Model:
            def __init__(self, args):
                pass

            def parameters(self):
                return {"expected.weight": object()}

            def load_weights(self, items, strict=False):
                raise AssertionError("load_weights must not run without mapped items")

            def eval(self):
                raise AssertionError("text model must not eval with unmatched weights")

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {"unmapped.weight": object()}
        gemma_module = types.ModuleType("mlx_lm.models.gemma3_text")
        gemma_module.ModelArgs = FakeModelArgs
        gemma_module.Gemma3Model = FakeGemma3Model
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, gemma_module.__name__, gemma_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe.text_proj = lambda pooled: pooled  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="text encoder weights did not match"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, in_features=16)

    def test_ltx23_text_encoder_streams_mapped_weight_filter(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)
        loaded_weights = []

        class GuardedItems:
            def __iter__(self):
                yield ("language_model.model.expected.weight", object())
                yield ("model.expected.bias", object())
                yield ("language_model.model.unmatched.weight", object())
                yield ("ignored.weight", object())

            def __len__(self):
                raise AssertionError("text encoder weights must not be list-materialized before load_weights")

        class FakeWeights:
            def items(self):
                return GuardedItems()

        class FakeTensor:
            def reshape(self, *args):
                return self

        class FakeHidden:
            def mean(self, axis):
                return "pooled"

        fake_mx = types.SimpleNamespace(
            array=lambda value: FakeTensor(),
            clear_cache=lambda: None,
            eval=lambda *args: None,
        )

        class FakeModelArgs:
            def __init__(self, **kwargs):
                pass

        class FakeGemma3Model:
            def __init__(self, args):
                pass

            def parameters(self):
                return {"expected.weight": object(), "expected.bias": object()}

            def load_weights(self, items, strict=False):
                assert strict is False
                assert not isinstance(items, list)
                loaded_weights.append(list(items))

            def eval(self):
                return None

            def __call__(self, input_ids):
                return FakeHidden()

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: FakeWeights()
        gemma_module = types.ModuleType("mlx_lm.models.gemma3_text")
        gemma_module.ModelArgs = FakeModelArgs
        gemma_module.Gemma3Model = FakeGemma3Model
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, gemma_module.__name__, gemma_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe.text_proj = lambda pooled: "context"  # type: ignore[method-assign]

        assert pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, in_features=16) == "context"
        assert [[key for key, _value in loaded] for loaded in loaded_weights] == [
            ["expected.weight", "expected.bias"]
        ]

    def test_ltx23_text_encoder_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        _install_fake_transformers_tokenizer(monkeypatch)

        fake_mx = types.SimpleNamespace(
            clear_cache=lambda: None,
            eval=lambda *args: None,
        )

        class FakeModelArgs:
            def __init__(self, **kwargs):
                pass

        class FakeGemma3Model:
            def __init__(self, args):
                raise RuntimeError("gemma construct failed after runtime opened")

        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {"model.weight": object()}
        gemma_module = types.ModuleType("mlx_lm.models.gemma3_text")
        gemma_module.ModelArgs = FakeModelArgs
        gemma_module.Gemma3Model = FakeGemma3Model
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, gemma_module.__name__, gemma_module)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe.text_proj = lambda pooled: pooled  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="gemma construct failed"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, in_features=16)

        assert cleanup_calls == ["cleanup"]

    def test_wan22_latent_init_preflights_tensor_allocation(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(AssertionError("latent allocation must be preflighted")),
            ),
            eval=lambda *args: None,
        )
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_host_allocation = lambda required_bytes, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("wan latent too large")
        )

        with pytest.raises(RuntimeMemoryAbort, match="wan latent too large"):
            pipe.init_latents(seed=1, width=256, height=256, frames=4)

    def test_wan22_latent_init_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(RuntimeError("wan latent init failed")),
            ),
            eval=lambda *args: None,
        )
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan latent init failed"):
            pipe.init_latents(seed=1, width=256, height=256, frames=4)

        assert cleanup_calls == ["cleanup"]

    def test_wan22_latent_init_rejects_shape_args_that_bypass_pipeline_shape(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(
            random=types.SimpleNamespace(
                seed=lambda seed: None,
                normal=lambda shape: (_ for _ in ()).throw(AssertionError("latent allocation must be rejected first")),
            ),
            eval=lambda *args: None,
        )
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="latent_init shape .* pipeline shape"):
            pipe.init_latents(seed=1, width=8192, height=8192, frames=64)

    def test_wan22_denoise_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeLatents:
            shape = (48, 1, 16, 16)

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("wan denoise failed")),
        )
        pipe.model = object()
        pipe.scheduler = types.SimpleNamespace(timesteps={0: 1})
        pipe.latent_shape = (48, 1, 16, 16)
        pipe.seq_len = 1
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()
        pipe.context_cond = object()
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan denoise failed"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

        assert cleanup_calls == ["cleanup"]

    def test_text_and_decode_reuse_is_blocked_to_prevent_metal_accumulation(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        ltx.model = object()
        wan.mx = object()
        wan.config = object()
        wan.model = object()
        wan.t5_encoder = object()
        wan.tokenizer = object()
        wan.seq_len = 1
        ltx._text_encode_started = True
        ltx._decode_started = True
        wan._text_encode_started = True
        wan._decode_started = True

        ltx.config = types.SimpleNamespace(caption_channels=16, cross_attention_dim=16, in_channels=128)
        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            ltx.encode_text({"prompt": "prompt", "negative_prompt": ""})
        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            ltx.decode(object())
        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            wan.encode_text({"prompt": "prompt", "negative_prompt": ""})
        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            wan.decode(object())

    def test_wan22_denoise_preflights_tensor_floor_before_mx_array(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeLatents:
            shape = (16, 1, 32, 32)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            guidance=3.0,
        )
        pipe.mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mx.array must be preflighted")),
        )
        pipe.model = object()
        pipe.scheduler = types.SimpleNamespace(timesteps={0: 1})
        pipe.latent_shape = (16, 1, 32, 32)
        pipe.seq_len = 1
        pipe.context_cfg = object()
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("denoise tensors too large")
        )

        with pytest.raises(RuntimeMemoryAbort, match="denoise tensors too large"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=3.0, cache="none")

    def test_wan22_denoise_rejects_latent_shape_mismatch_before_mx_array(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeLatents:
            shape = (48, 99, 1024, 1024)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            guidance=3.0,
        )
        pipe.mx = types.SimpleNamespace(
            array=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mx.array must be rejected first")),
        )
        pipe.model = object()
        pipe.scheduler = types.SimpleNamespace(timesteps=types.SimpleNamespace(tolist=lambda: [1]))
        pipe.latent_shape = (48, 1, 16, 16)
        pipe.seq_len = 1
        pipe.context_cfg = object()
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="latent shape .* expected"):
            pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=3.0, cache="none")

    def test_ltx23_text_encoder_rejects_token_sequence_before_mlx_array(self, tmp_path, monkeypatch):
        import numpy as np
        import fastgen_profiler.mlx_guard as mlx_guard
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "model_type": "gemma3",
                        "hidden_size": 16,
                        "num_hidden_layers": 1,
                        "intermediate_size": 32,
                        "num_attention_heads": 1,
                        "head_dim": 16,
                        "rms_norm_eps": 1e-6,
                        "vocab_size": 32,
                        "num_key_value_heads": 1,
                        "rope_theta": 10000,
                        "sliding_window": 8,
                        "sliding_window_pattern": 1,
                        "max_position_embeddings": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")

        fake_mx = types.SimpleNamespace(
            array=lambda value: (_ for _ in ()).throw(AssertionError("mx.array must be preflighted first")),
            clear_cache=lambda: None,
            eval=lambda *args: None,
        )

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": np.array([[1, 2, 3]])}

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path, local_files_only):
                return FakeTokenizer()

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        utils_module = types.ModuleType("mlx_video.models.ltx_2.utils")
        utils_module.load_safetensors = lambda path: {}
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"}:
                raise AssertionError("MLX import must be rejected before token budget")
            if name == "mlx_lm.models.gemma3_text":
                raise AssertionError("Gemma text model import must be rejected before token budget")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "transformers", transformers_module)
        monkeypatch.setitem(sys.modules, "mlx_video.models.ltx_2.utils", utils_module)
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", None)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True

        with pytest.raises(MemoryGuardError, match="token sequence is 3 tokens"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, in_features=16)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("max_position_embeddings", "2", "max_position_embeddings=.*positive structural dimension"),
            ("max_position_embeddings", 2.5, "max_position_embeddings=.*positive structural dimension"),
            ("hidden_size", "16", "hidden_size=.*positive structural dimension"),
            ("head_dim", "16", "head_dim=.*positive structural dimension"),
            ("num_key_value_heads", 1.5, "num_key_value_heads=.*positive structural dimension"),
            ("sliding_window", "8", "sliding_window=.*positive structural dimension"),
            ("sliding_window_pattern", 0, "sliding_window_pattern=.*positive structural dimension"),
        ],
    )
    def test_ltx23_text_encoder_rejects_non_integer_token_budget_config(
        self, tmp_path, monkeypatch, field, value, message
    ):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        config_path = text_encoder_dir / "config.json"
        full_config = json.loads(config_path.read_text(encoding="utf-8"))
        full_config["text_config"][field] = value
        config_path.write_text(json.dumps(full_config), encoding="utf-8")
        _install_fake_transformers_tokenizer(monkeypatch, token_count=1)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match=message):
            pipe._preflight_text_prompt_tokens("prompt", text_encoder_dir, tokenizer_dir)

    @pytest.mark.parametrize(
        "field",
        [
            "head_dim",
            "num_key_value_heads",
            "sliding_window",
            "sliding_window_pattern",
        ],
    )
    def test_ltx23_text_encoder_rejects_oversized_structural_config(
        self,
        tmp_path,
        monkeypatch,
        field,
    ):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        _write_ltx_text_encoder_fixture(text_encoder_dir, tokenizer_dir)
        config_path = text_encoder_dir / "config.json"
        full_config = json.loads(config_path.read_text(encoding="utf-8"))
        full_config["text_config"][field] = 65_537
        config_path.write_text(json.dumps(full_config), encoding="utf-8")
        _install_fake_transformers_tokenizer(monkeypatch, token_count=1)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match=rf"{field}=65537 exceeds safe structural dimension"):
            pipe._preflight_text_prompt_tokens("prompt", text_encoder_dir, tokenizer_dir)

    def test_wan22_text_encoder_rejects_token_sequence_before_external_encode(self, tmp_path, monkeypatch):
        import numpy as np
        import fastgen_profiler.mlx_guard as mlx_guard
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": np.array([[1, 2, 3]])}

        def encode_text(*args, **kwargs):
            raise AssertionError("mlx_video encode_text must be preflighted first")

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.encode_text = encode_text
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        monkeypatch.setattr(mlx_guard, "_current_mlx_memory_limit_bytes", None)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            guidance=1.0,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(
            text_len=2,
            dim=16,
            patch_size=(1, 1, 1),
            sample_neg_prompt="",
        )
        pipe.model = object()
        pipe.t5_encoder = object()
        pipe.tokenizer = FakeTokenizer()
        pipe.seq_len = 1
        pipe.latent_shape = (16, 1, 1, 1)

        with pytest.raises(MemoryGuardError, match="token sequence is 3 tokens"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("text_len", "128", "text_len=.*positive integer"),
            ("text_len", 128.5, "text_len=.*positive integer"),
            ("dim", "4096", "dim=.*positive integer"),
        ],
    )
    def test_wan22_text_encoder_rejects_non_integer_token_budget_config(self, tmp_path, field, value, message):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": [[1]]}

        config_values = {
            "text_len": 128,
            "dim": 4096,
            "patch_size": (1, 1, 1),
            "sample_neg_prompt": "",
        }
        config_values[field] = value
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            guidance=1.0,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(**config_values)
        pipe.model = object()
        pipe.t5_encoder = object()
        pipe.tokenizer = FakeTokenizer()
        pipe.seq_len = 1
        pipe.latent_shape = (16, 1, 1, 1)

        with pytest.raises(RuntimeMemoryAbort, match=message):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_wan22_text_encoder_preflights_rope_tensors_before_external_encode(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        def encode_text(*args, **kwargs):
            raise AssertionError("external encode_text must be rejected first")

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.encode_text = encode_text
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": [[1]]}

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None)
        pipe.config = types.SimpleNamespace(
            text_len=128,
            dim=4096,
            patch_size=(1, 1, 1),
        )
        pipe.model = object()
        pipe.t5_encoder = object()
        pipe.tokenizer = FakeTokenizer()
        pipe.seq_len = 1
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]

        def check_floor(elements, phase, multiplier=4):
            if phase == "text_encoder rope tensors":
                raise RuntimeMemoryAbort("rope tensors too large")
            return None

        pipe._check_mlx_tensor_floor = check_floor  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match="rope tensors too large"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

    def test_wan22_encode_text_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        def encode_text(*args, **kwargs):
            raise RuntimeError("wan text encode failed after runtime opened")

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.encode_text = encode_text
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                return {"input_ids": [[1]]}

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            guidance=1.0,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(text_len=128, dim=4096, patch_size=(1, 1, 1))
        pipe.model = object()
        pipe.t5_encoder = object()
        pipe.tokenizer = FakeTokenizer()
        pipe.seq_len = 1
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan text encode failed"):
            pipe.encode_text({"prompt": "prompt", "negative_prompt": ""})

        assert cleanup_calls == ["cleanup"]

    def test_load_model_reuse_is_blocked_to_prevent_metal_accumulation(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        ltx.model = object()
        wan.model = object()

        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            ltx.load_model()
        with pytest.raises(RuntimeError, match="fresh pipeline/process"):
            wan.load_model()

    def test_wan22_denoise_checks_memory_before_work(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeLatents:
            shape = (16, 1, 1, 1)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = object()
        pipe.model = object()
        pipe.scheduler = object()
        pipe.latent_shape = (16, 1, 1, 1)
        pipe.seq_len = 1
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()

        with patch(
            "fastgen_profiler.mlx_guard.check_runtime_memory",
            side_effect=RuntimeMemoryAbort("stop before wan work"),
        ):
            with pytest.raises(RuntimeMemoryAbort, match="stop before wan work"):
                pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none")

    @pytest.mark.parametrize(
        ("step_index", "steps"),
        [
            (-1, 1),
            (1, 1),
            (0, 0),
            (0, 513),
            (True, 1),
            (0, True),
        ],
    )
    def test_wan22_denoise_rejects_invalid_step_arguments_before_work(
        self,
        tmp_path,
        monkeypatch,
        step_index,
        steps,
    ):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeLatents:
            shape = (16, 1, 1, 1)

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = object()
        pipe.model = object()
        pipe.scheduler = object()
        pipe.latent_shape = (16, 1, 1, 1)
        pipe.seq_len = 1
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()
        pipe._check_memory = lambda phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("memory check must not run for invalid denoise arguments")
        )

        with pytest.raises(RuntimeMemoryAbort, match="Wan2.2 denoise step arguments"):
            pipe.denoise_step(FakeLatents(), step_index=step_index, steps=steps, guidance=1.0, cache="none")

        assert cleanup_calls == ["cleanup"]

    def test_wan22_denoise_indexes_scheduler_timestep_without_materializing_all_timesteps(
        self,
        tmp_path,
    ):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeTimesteps:
            def __getitem__(self, index):
                assert index == 0
                return 0.5

            def tolist(self):
                raise AssertionError("denoise_step must not materialize all scheduler timesteps")

        class FakeScheduler:
            timesteps = FakeTimesteps()

            def step(self, noise_pred, timestep_val, latents):
                return types.SimpleNamespace(squeeze=lambda axis: "next_latents")

        class FakeMX:
            def eval(self, *args):
                return None

            def array(self, value):
                return value

        class FakeLatents:
            shape = (16, 1, 1, 1)

            def __getitem__(self, key):
                return self

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return [FakeLatents()]

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = FakeMX()
        pipe.model = FakeModel()
        pipe.scheduler = FakeScheduler()
        pipe.latent_shape = (16, 1, 1, 1)
        pipe.seq_len = 1
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()
        pipe.context_cond = object()
        pipe.cfg_disabled = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_mlx_tensor_floor = lambda elements, phase, multiplier=4: None  # type: ignore[method-assign]

        assert pipe.denoise_step(FakeLatents(), step_index=0, steps=1, guidance=1.0, cache="none") == "next_latents"

    def test_ltx23_denoise_checks_memory_before_work(self, tmp_path):
        pytest.importorskip("mlx_video.models.ltx_2.config")
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()

        with patch(
            "fastgen_profiler.mlx_guard.check_runtime_memory",
            side_effect=RuntimeMemoryAbort("stop before ltx work"),
        ):
            with pytest.raises(RuntimeMemoryAbort, match="stop before ltx work"):
                pipe.denoise_step(object(), step_index=0, steps=1, guidance=1.0, cache="none")

    @pytest.mark.parametrize(
        ("step_index", "steps"),
        [
            (-1, 1),
            (1, 1),
            (0, 0),
            (0, 513),
            (True, 1),
            (0, True),
        ],
    )
    def test_ltx23_denoise_rejects_invalid_step_arguments_before_work(
        self,
        tmp_path,
        monkeypatch,
        step_index,
        steps,
    ):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class FakeLatents:
            dtype = "float32"
            shape = (1, 128, 4, 32, 32)

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._check_memory = lambda phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("memory check must not run for invalid denoise arguments")
        )

        with pytest.raises(RuntimeMemoryAbort, match="LTX2.3 denoise step arguments"):
            pipe.denoise_step(FakeLatents(), step_index=step_index, steps=steps, guidance=1.0, cache="none")

        assert cleanup_calls == ["cleanup"]

    def test_wan22_file_preflight_uses_file_size_before_loading(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        path = tmp_path / "model.safetensors"
        path.write_bytes(b"x" * 1024)
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch(
            "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
            side_effect=RuntimeMemoryAbort("file too large"),
        ) as guard:
            with pytest.raises(RuntimeMemoryAbort, match="file too large"):
                pipe._check_file_load(path, "read model")

        guard.assert_called_once_with(2048, label="wan2.2 read model")

    def test_ltx23_file_preflight_uses_file_size_before_loading(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        path = tmp_path / "model.safetensors"
        path.write_bytes(b"x" * 2048)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch(
            "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
            side_effect=RuntimeMemoryAbort("file too large"),
        ) as guard:
            with pytest.raises(RuntimeMemoryAbort, match="file too large"):
                pipe._check_file_load(path, "read shard")

        guard.assert_called_once_with(4096, label="ltx2.3 read shard")

    @pytest.mark.parametrize(
        ("elements", "multiplier", "message"),
        [
            (0, 4, "elements must be a positive integer"),
            (-1, 4, "elements must be a positive integer"),
            (True, 4, "elements must be a positive integer"),
            (1.5, 4, "elements must be a positive integer"),
            (16, 0, "multiplier must be a positive integer"),
            (16, False, "multiplier must be a positive integer"),
            (16, 2.5, "multiplier must be a positive integer"),
        ],
    )
    def test_wan22_mlx_tensor_floor_rejects_invalid_inputs(self, tmp_path, elements, multiplier, message):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(MemoryGuardError, match=message):
            pipe._check_mlx_tensor_floor(elements, "tensor floor", multiplier=multiplier)  # type: ignore[arg-type]

    def test_file_preflight_fails_closed_when_stat_unavailable(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        missing = tmp_path / "missing.safetensors"
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="cannot stat"):
            ltx._check_file_load(missing, "read missing")
        with pytest.raises(RuntimeMemoryAbort, match="cannot stat"):
            wan._check_file_load(missing, "read missing")

    def test_ltx23_directory_preflight_sums_safetensors(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_dir = tmp_path / "vae"
        model_dir.mkdir()
        (model_dir / "a.safetensors").write_bytes(b"x" * 10)
        (model_dir / "b.safetensors").write_bytes(b"x" * 20)
        (model_dir / "ignore.bin").write_bytes(b"x" * 100)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch("fastgen_profiler.mlx_guard.check_host_allocation_headroom") as guard:
            pipe._check_directory_load(model_dir, "read vae")

        guard.assert_called_once_with(60, label="ltx2.3 read vae")

    def test_ltx23_directory_preflight_fails_closed_when_missing(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="cannot scan"):
            pipe._check_directory_load(tmp_path / "missing-dir", "read missing")

    def test_ltx23_directory_preflight_rejects_excessive_recursive_scan(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        model_dir = tmp_path / "vae"
        nested = model_dir / "nested"
        nested.mkdir(parents=True)
        (nested / "model.safetensors").write_bytes(b"x")
        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PRELOAD_SCAN_DIRS", 1)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="directory scan exceeded 1 directories"):
            pipe._check_directory_load(model_dir, "read vae")

    def test_ltx23_flat_shard_listing_rejects_excessive_file_scan(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder = tmp_path / "text"
        text_encoder.mkdir()
        (text_encoder / "config.json").write_text("{}", encoding="utf-8")
        (text_encoder / "a.safetensors").write_bytes(b"x")
        (text_encoder / "b.safetensors").write_bytes(b"x")
        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PRELOAD_SCAN_FILES", 1)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="file scan exceeded 1 files"):
            pipe._preflight_text_encoder_assets(text_encoder)

    def test_tokenizer_preflight_checks_file_size_before_loading(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_bytes(b"x" * 100)
        (tokenizer_dir / "tokenizer_config.json").write_bytes(b"x" * 25)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch("fastgen_profiler.mlx_guard.check_host_allocation_headroom") as guard:
            ltx._check_tokenizer_load(tokenizer_dir, "read tokenizer")
            wan._check_tokenizer_load(tokenizer_dir, "read tokenizer")

        assert guard.call_args_list[0].args == (500,)
        assert guard.call_args_list[0].kwargs == {"label": "ltx2.3 read tokenizer"}
        assert guard.call_args_list[1].args == (500,)
        assert guard.call_args_list[1].kwargs == {"label": "wan2.2 read tokenizer"}

    def test_tokenizer_preflight_rejects_excessive_file_scan(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter, wan22_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_bytes(b"x")
        (tokenizer_dir / "tokenizer_config.json").write_bytes(b"x")
        monkeypatch.setattr(ltx23_mlx_adapter, "_MAX_PRELOAD_SCAN_FILES", 1)
        monkeypatch.setattr(wan22_mlx_adapter, "_MAX_PRELOAD_SCAN_FILES", 1)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="file scan exceeded 1 files"):
            ltx._check_tokenizer_load(tokenizer_dir, "read tokenizer")
        with pytest.raises(RuntimeMemoryAbort, match="file scan exceeded 1 files"):
            wan._check_tokenizer_load(tokenizer_dir, "read tokenizer")

    def test_ltx23_text_encoder_preflight_runs_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        calls: list[str] = []
        pipe._check_directory_load = lambda path, phase: calls.append(phase) or (_ for _ in ()).throw(
            RuntimeMemoryAbort("text encoder too large")
        )  # type: ignore[method-assign]
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text"}:
                raise AssertionError("text encoder preflight must run before MLX/Gemma imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="text encoder too large"):
            pipe._encode_with_gemma3("prompt", tmp_path / "text_encoder", tmp_path / "tokenizer", 4096)

        assert calls == ["preflight text_encoder"]

    def test_ltx23_text_encoder_missing_config_blocks_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text"}:
                raise AssertionError("text encoder config check must run before MLX/Gemma imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(FileNotFoundError, match="text encoder config not found"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, 4096)

    def test_ltx23_text_encoder_missing_weights_blocks_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(
            json.dumps({"text_config": {"hidden_size": 4096}}),
            encoding="utf-8",
        )
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text"}:
                raise AssertionError("text encoder weights check must run before MLX/Gemma imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(FileNotFoundError, match="text encoder weights not found"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, 4096)

    def test_ltx23_text_encoder_model_config_preflight_runs_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "hidden_size": 1_000_000_000,
                        "intermediate_size": 4096,
                        "num_hidden_layers": 1,
                        "vocab_size": 1024,
                        "num_attention_heads": 1,
                    }
                }
            ),
            encoding="utf-8",
        )
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text"}:
                raise AssertionError("text config preflight must run before MLX/Gemma imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="hidden_size=1000000000 exceeds safe structural dimension"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, 4096)

    def test_ltx23_text_encoder_config_file_size_preflight_runs_before_json_read(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text("{not-json}", encoding="utf-8")
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort(f"{phase} too large")
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text"}:
                raise AssertionError("text config file preflight must run before MLX/Gemma imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="read text_encoder config too large"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, 4096)

    def test_ltx23_text_encoder_config_rejects_oversized_json_before_tokenizer(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_mlx_adapter
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        text_encoder_dir = tmp_path / "text_encoder"
        tokenizer_dir = tmp_path / "tokenizer"
        text_encoder_dir.mkdir()
        tokenizer_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(
            " " * (ltx23_mlx_adapter._MAX_CONFIG_JSON_BYTES + 1),
            encoding="utf-8",
        )
        (text_encoder_dir / "model.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_directory_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core", "mlx_lm.models.gemma3_text", "transformers"}:
                raise AssertionError("oversized text config must be rejected before tokenizer or MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="above safe config limit"):
            pipe._encode_with_gemma3("prompt", text_encoder_dir, tokenizer_dir, 4096)

    def test_wan22_tokenizer_preflight_blocks_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "t5_encoder.safetensors").write_bytes(b"x")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_bytes(b"x" * 1024)
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        def abort_tokenizer_only(required_bytes, *, label):
            if label == "wan2.2 preflight tokenizer":
                raise RuntimeMemoryAbort("tokenizer too large")
            return None

        with (
            patch(
                "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
                side_effect=abort_tokenizer_only,
            ),
            patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits,
        ):
            with pytest.raises(RuntimeMemoryAbort, match="tokenizer too large"):
                pipe.load_model()

        limits.assert_not_called()

    def test_ltx23_decode_missing_vae_blocks_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("VAE existence check must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(FileNotFoundError, match="No VAE decoder found"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_missing_vae_weights_blocks_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("VAE weight preflight must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(FileNotFoundError, match="VAE decoder weights not found"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_preflights_vae_before_mlx_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")
        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._check_directory_load = lambda path, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("vae too large")
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("VAE preflight must run before MLX imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(RuntimeMemoryAbort, match="vae too large"):
            pipe.decode(FakeLatents())

    def test_wan22_decode_preflights_vae_file_before_mlx_video_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._check_file_load = lambda path, phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeMemoryAbort("vae file too large")
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise AssertionError("Wan VAE preflight must run before mlx_video imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        class FakeLatents:
            shape = (48, 1, 16, 16)

        with pytest.raises(RuntimeMemoryAbort, match="vae file too large"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_rejects_latent_shape_before_vae_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")

        class FakeLatents:
            shape = (1, 128, 999, 32, 32)

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("latent shape validation must run before LTX VAE imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)

        with pytest.raises(RuntimeError, match="latent shape .* expected"):
            pipe.decode(FakeLatents())

    def test_wan22_decode_rejects_latent_shape_before_vae_import(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "vae.safetensors").write_bytes(b"x")

        class FakeLatents:
            shape = (48, 999, 16, 16)

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise AssertionError("latent shape validation must run before Wan VAE imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)

        with pytest.raises(RuntimeError, match="latent shape .* expected"):
            pipe.decode(FakeLatents())

    def test_ltx23_upsampler_temp_vae_preflight_runs_before_loading(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").write_bytes(b"x")

        fake_mx = types.SimpleNamespace(eval=lambda *args: None)

        class FakeUpsampler:
            def parameters(self):
                return object()

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                raise AssertionError("VAE temp load must be preflighted first")

        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (FakeUpsampler(), None)
        upsampler_module.upsample_latents = lambda *args: object()
        for name in [
            "mlx_video",
            "mlx_video.models",
            "mlx_video.models.ltx_2",
            "mlx_video.models.ltx_2.video_vae",
        ]:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        calls: list[str] = []
        pipe._check_memory = lambda phase: calls.append(f"memory:{phase}")  # type: ignore[method-assign]
        pipe._check_file_load = lambda path, phase: calls.append(f"file:{phase}")  # type: ignore[method-assign]

        def check_directory(path, phase):
            calls.append(f"dir:{phase}")
            if phase == "read upsampler_vae_stats":
                raise RuntimeMemoryAbort("temp vae too large")

        pipe._check_directory_load = check_directory  # type: ignore[method-assign]

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(RuntimeMemoryAbort, match="temp vae too large"):
            pipe.decode(FakeLatents())

        assert "file:preflight upsampler" in calls
        assert "dir:read upsampler_vae_stats" in calls

    def test_ltx23_decode_preflights_output_tensor_before_vae_forward(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, latents):
                raise AssertionError("VAE forward must be preflighted first")

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (object(), None)
        upsampler_module.upsample_latents = lambda *args: object()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (
            (_ for _ in ()).throw(RuntimeMemoryAbort("decode output too large"))
            if phase == "vae output tensor"
            else None
        )  # type: ignore[method-assign]

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(RuntimeMemoryAbort, match="decode output too large"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, latents):
                raise RuntimeError("vae failed after runtime opened")

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (object(), None)
        upsampler_module.upsample_latents = lambda *args: object()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(RuntimeError, match="vae failed"):
            pipe.decode(FakeLatents())

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_decode_runtime_abort_from_upsampled_shape_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").write_bytes(b"x")

        class FakeVAE:
            per_channel_statistics = types.SimpleNamespace(mean=0, std=1)

            def parameters(self):
                return object()

            def __call__(self, latents):
                raise AssertionError("VAE forward must not run after invalid upsampled latent shape")

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        class FakeUpsampledLatents:
            shape = (1, 128, 4, "64", 64)

        class FakeUpsampler:
            def parameters(self):
                return object()

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (FakeUpsampler(), None)
        upsampler_module.upsample_latents = lambda *args: FakeUpsampledLatents()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)

        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match="non-integer dimension"):
            pipe.decode(FakeLatents())

        assert cleanup_calls == ["cleanup"]

    def test_ltx23_decode_rejects_unexpected_frame_shape_before_numpy(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, latents):
                return object()

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

        fake_mx = types.SimpleNamespace(
            eval=lambda *args: None,
            squeeze=lambda video, axis: FakeTensor((3, 5, 256, 256)),
            transpose=lambda video, axes: FakeTensor((5, 256, 256, 3)),
            clear_cache=lambda: None,
        )
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (object(), None)
        upsampler_module.upsample_latents = lambda *args: object()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter._numpy",
            lambda: types.SimpleNamespace(
                array=lambda value: (_ for _ in ()).throw(AssertionError("np.array must be rejected first"))
            ),
        )

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        with pytest.raises(RuntimeError, match="decoded LTX2.3 frames must have shape"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_preflights_upsampled_latents_before_vae_forward(self, tmp_path, monkeypatch):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").write_bytes(b"x")

        class FakeVAE:
            per_channel_statistics = types.SimpleNamespace(mean=0, std=1)

            def parameters(self):
                return object()

            def __call__(self, latents):
                raise AssertionError("VAE forward must be rejected before oversized upsampled latents")

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        class FakeUpsampledLatents:
            shape = (1, 128, 4, 64, 64)

        class FakeUpsampler:
            def parameters(self):
                return object()

        fake_mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (FakeUpsampler(), None)
        upsampler_module.upsample_latents = lambda *args: FakeUpsampledLatents()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (
            (_ for _ in ()).throw(RuntimeMemoryAbort("upsampled latent tensor too large"))
            if phase == "upsampled latent tensor"
            else None
        )

        with pytest.raises(RuntimeMemoryAbort, match="upsampled latent tensor too large"):
            pipe.decode(FakeLatents())

    def test_ltx23_tensor_allocation_rejects_coerced_shape_dimensions(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class FakeTensor:
            shape = (1, "128", 4, 64, 64)

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="non-integer dimension"):
            pipe._check_tensor_shape_allocation(FakeTensor(), "coerced tensor")

    def test_ltx23_tensor_allocation_rejects_unbounded_shape_metadata(self, tmp_path):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        class Shape:
            def __iter__(self):
                yield from (1, 128, 4, 64, 64, 1)
                raise AssertionError("tensor allocation preflight must stop at expected rank")

        class FakeTensor:
            shape = Shape()

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with pytest.raises(RuntimeMemoryAbort, match="shape rank exceeds"):
            pipe._check_tensor_shape_allocation(FakeTensor(), "unbounded tensor")

    def test_wan22_decode_preflights_output_tensor_before_vae_forward(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "vae.safetensors").write_bytes(b"x")

        class FakeLatents:
            shape = (48, 1, 16, 16)

            def transpose(self, *args):
                return self

            def __getitem__(self, key):
                return self

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, z):
                raise AssertionError("Wan VAE forward must be preflighted first")

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_vae_decoder = lambda path, config: FakeVAE()
        vae_module = types.ModuleType("mlx_video.models.wan_2.vae22")
        vae_module.denormalize_latents = lambda z: z
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.vae22", vae_module)

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: (
            (_ for _ in ()).throw(RuntimeMemoryAbort("wan decode output too large"))
            if phase == "vae output tensor"
            else None
        )  # type: ignore[method-assign]

        with pytest.raises(RuntimeMemoryAbort, match="wan decode output too large"):
            pipe.decode(FakeLatents())

    def test_wan22_decode_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "vae.safetensors").write_bytes(b"x")

        class FakeLatents:
            shape = (48, 1, 16, 16)

            def transpose(self, *args):
                return self

            def __getitem__(self, key):
                return self

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, z):
                raise RuntimeError("wan vae failed after runtime opened")

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_vae_decoder = lambda path, config: FakeVAE()
        vae_module = types.ModuleType("mlx_video.models.wan_2.vae22")
        vae_module.denormalize_latents = lambda z: z
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.vae22", vae_module)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan vae failed"):
            pipe.decode(FakeLatents())

        assert cleanup_calls == ["cleanup"]

    def test_wan22_decode_rejects_unexpected_frame_shape_before_numpy(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "vae.safetensors").write_bytes(b"x")

        class FakeLatents:
            shape = (48, 1, 16, 16)

            def transpose(self, *args):
                return self

            def __getitem__(self, key):
                return self

        class FakeFrame:
            shape = (5, 256, 256, 3)

        class FakeVideo:
            def __getitem__(self, key):
                return FakeFrame()

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, z):
                return FakeVideo()

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_vae_decoder = lambda path, config: FakeVAE()
        vae_module = types.ModuleType("mlx_video.models.wan_2.vae22")
        vae_module.denormalize_latents = lambda z: z
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.vae22", vae_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter._numpy",
            lambda: types.SimpleNamespace(
                array=lambda value: (_ for _ in ()).throw(AssertionError("np.array must be rejected first"))
            ),
        )

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="decoded Wan2.2 frames must have shape"):
            pipe.decode(FakeLatents())

    def test_ltx23_decode_numpy_frame_preflight_reuses_validated_shape(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        vae_decoder_dir = tmp_path / "vae" / "decoder"
        vae_decoder_dir.mkdir(parents=True)
        (vae_decoder_dir / "decoder.safetensors").write_bytes(b"x")

        class SinglePassShape:
            def __init__(self, dims):
                self._dims = dims
                self._iterations = 0

            def __iter__(self):
                self._iterations += 1
                if self._iterations > 1:
                    raise AssertionError("decode must not re-read frame shape after validation")
                yield from self._dims

        class FakeFrame:
            def __init__(self):
                self.shape = SinglePassShape((4, 256, 256, 3))

            def astype(self, *_args, **_kwargs):
                return self

        frame = FakeFrame()

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, latents):
                return object()

        class FakeVideoDecoder:
            @classmethod
            def from_pretrained(cls, path):
                return FakeVAE()

        fake_mx = types.SimpleNamespace(
            eval=lambda *args: None,
            squeeze=lambda video, axis: object(),
            transpose=lambda video, axes: frame,
            clear_cache=lambda: None,
        )
        decoder_module = types.ModuleType("mlx_video.models.ltx_2.video_vae.decoder")
        decoder_module.VideoDecoder = FakeVideoDecoder
        upsampler_module = types.ModuleType("mlx_video.models.ltx_2.upsampler")
        upsampler_module.load_upsampler = lambda path: (object(), None)
        upsampler_module.upsample_latents = lambda *args: object()
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
        monkeypatch.setitem(sys.modules, upsampler_module.__name__, upsampler_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter._numpy",
            lambda: types.SimpleNamespace(
                array=lambda value: value,
                add=lambda *args, **kwargs: None,
                multiply=lambda *args, **kwargs: None,
                clip=lambda *args, **kwargs: None,
                uint8="uint8",
            ),
        )

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        assert pipe.decode(FakeLatents()) is frame

    def test_wan22_decode_numpy_frame_preflight_reuses_validated_shape(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "vae.safetensors").write_bytes(b"x")

        class SinglePassShape:
            def __init__(self, dims):
                self._dims = dims
                self._iterations = 0

            def __iter__(self):
                self._iterations += 1
                if self._iterations > 1:
                    raise AssertionError("decode must not re-read frame shape after validation")
                yield from self._dims

        class FakeLatents:
            shape = (48, 1, 16, 16)

            def transpose(self, *args):
                return self

            def __getitem__(self, key):
                return self

        class FakeFrame:
            def __init__(self):
                self.shape = SinglePassShape((4, 256, 256, 3))

            def astype(self, *_args, **_kwargs):
                return self

        frame = FakeFrame()

        class FakeVideo:
            def __getitem__(self, key):
                return frame

        class FakeVAE:
            def parameters(self):
                return object()

            def __call__(self, z):
                return FakeVideo()

        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_vae_decoder = lambda path, config: FakeVAE()
        vae_module = types.ModuleType("mlx_video.models.wan_2.vae22")
        vae_module.denormalize_latents = lambda z: z
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.utils", utils_module)
        monkeypatch.setitem(sys.modules, "mlx_video.models.wan_2.vae22", vae_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter._numpy",
            lambda: types.SimpleNamespace(
                array=lambda value: value,
                add=lambda *args, **kwargs: None,
                multiply=lambda *args, **kwargs: None,
                clip=lambda *args, **kwargs: None,
                uint8="uint8",
            ),
        )

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.mx = types.SimpleNamespace(eval=lambda *args: None, clear_cache=lambda: None)
        pipe.config = types.SimpleNamespace(vae_z_dim=48)
        pipe.latent_shape = (48, 1, 16, 16)
        pipe._mlx_runtime_ready = True
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]

        assert pipe.decode(FakeLatents()) is frame

    @pytest.mark.parametrize("adapter", ["ltx", "wan"])
    def test_adapter_shape_validation_rejects_unbounded_shape_metadata(self, tmp_path, adapter):
        if adapter == "ltx":
            from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

            pipe = LTX23MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            )
            expected = (4, 256, 256, 3, 1)
            validator = pipe._validate_frame_shape
        else:
            from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

            pipe = Wan22MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            )
            expected = (4, 256, 256, 3, 1)
            validator = pipe._validate_frame_shape

        class Shape:
            def __iter__(self):
                yield from expected
                raise AssertionError("shape validator must not consume beyond expected rank")

        class FakeFrames:
            shape = Shape()

        with pytest.raises(RuntimeMemoryAbort, match="shape rank"):
            validator(FakeFrames(), "decode")

    @pytest.mark.parametrize(
        ("step_index", "steps", "message"),
        [
            (0, 0, "steps must be positive"),
            (0, 513, "exceeds safe maximum"),
            (-1, 1, "step_index=-1 must be in"),
            (1, 1, "step_index=1 must be in"),
            (False, 1, "step_index must be an integer"),
            (0, True, "steps must be an integer"),
        ],
    )
    def test_ltx23_denoise_rejects_invalid_step_arguments_before_mlx_import(
        self,
        tmp_path,
        monkeypatch,
        step_index,
        steps,
        message,
    ):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

        pipe = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe.model = object()
        pipe.config = types.SimpleNamespace(in_channels=128)

        class FakeLatents:
            shape = (1, 128, 4, 32, 32)

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"mlx", "mlx.core"} or name.startswith("mlx_video"):
                raise AssertionError("invalid denoise step arguments must be rejected before MLX import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match=message):
            pipe.denoise_step(FakeLatents(), step_index=step_index, steps=steps, guidance=1.0, cache="none")

    def test_video_encode_preflights_frame_buffer(self, tmp_path):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )

        with patch(
            "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
            side_effect=RuntimeMemoryAbort("encode too large"),
        ):
            with pytest.raises(RuntimeMemoryAbort, match="encode too large"):
                ltx.encode_video(frames, fps=24)
            with pytest.raises(RuntimeMemoryAbort, match="encode too large"):
                wan.encode_video(frames, fps=24)

    @pytest.mark.parametrize("adapter", ["ltx", "wan"])
    def test_video_encode_validates_shape_before_ndim_access(self, tmp_path, adapter):
        if adapter == "ltx":
            from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

            pipe = LTX23MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            )
        else:
            from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

            pipe = Wan22MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
            )

        class FakeFrames:
            shape = (4, 256, 256, 3)

            @property
            def ndim(self):
                raise AssertionError("encode_video must use bounded shape validation before ndim access")

        frames = FakeFrames()

        assert pipe.encode_video(frames, fps=24) is frames

    @pytest.mark.parametrize("method_name", ["encode_video", "write_output"])
    @pytest.mark.parametrize("adapter", ["ltx", "wan"])
    def test_video_output_preflight_reuses_validated_frame_shape(
        self,
        tmp_path,
        adapter,
        method_name,
        monkeypatch,
    ):
        if adapter == "ltx":
            from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline

            pipe = LTX23MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
                save_video=method_name == "encode_video",
            )
        else:
            from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

            pipe = Wan22MLXPipeline(
                model_path=tmp_path,
                seed=1,
                width=256,
                height=256,
                frames=4,
                steps=1,
                save_video=method_name == "encode_video",
            )

        class SinglePassShape:
            def __init__(self):
                self._iterations = 0

            def __iter__(self):
                self._iterations += 1
                if self._iterations > 1:
                    raise AssertionError("video output preflight must not re-read frame shape")
                yield from (4, 256, 256, 3)

        class FakeFrames:
            shape = SinglePassShape()
            nbytes = 0

        frames = FakeFrames()
        monkeypatch.setattr("fastgen_profiler.mlx_guard.check_runtime_memory", lambda label: None)
        monkeypatch.setattr(
            "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
            lambda required, *, label: (_ for _ in ()).throw(RuntimeMemoryAbort(f"{label} blocked")),
        )

        if method_name == "encode_video":
            with pytest.raises(RuntimeMemoryAbort, match="video_encode frames blocked"):
                pipe.encode_video(frames, fps=24)
        else:
            with pytest.raises(RuntimeMemoryAbort, match="file_write frames blocked"):
                pipe.write_output(frames, tmp_path / f"{adapter}-out", run_id="r1")

    def test_write_output_preflights_numpy_frame_buffer(self, tmp_path):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch(
            "fastgen_profiler.mlx_guard.check_host_allocation_headroom",
            side_effect=RuntimeMemoryAbort("write too large"),
        ):
            with pytest.raises(RuntimeMemoryAbort, match="write too large"):
                ltx.write_output(frames, tmp_path / "ltx-out", run_id="r1")
            with pytest.raises(RuntimeMemoryAbort, match="write too large"):
                wan.write_output(frames, tmp_path / "wan-out", run_id="r1")

    def test_video_postprocess_uses_conservative_frame_buffer_budget(self, tmp_path, monkeypatch):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        expected = 4 * 256 * 256 * 3 * 4 * 6
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        captures: list[tuple[str, int, str]] = []

        for name, pipe in (("ltx", ltx), ("wan", wan)):
            pipe._check_memory = lambda phase: None  # type: ignore[method-assign]

            def capture(required_bytes: int, phase: str, *, pipe_name: str = name) -> None:
                captures.append((pipe_name, required_bytes, phase))

            pipe._check_host_allocation = capture  # type: ignore[method-assign]

        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )

        for action in (
            lambda: ltx.encode_video(frames, fps=24),
            lambda: wan.encode_video(frames, fps=24),
            lambda: ltx.write_output(frames, tmp_path / "ltx-out", run_id="r1"),
            lambda: wan.write_output(frames, tmp_path / "wan-out", run_id="r1"),
        ):
            with pytest.raises(ModuleNotFoundError, match="before initializing MLX"):
                action()

        assert captures == [
            ("ltx", expected, "video_encode frames"),
            ("wan", expected, "video_encode frames"),
            ("ltx", expected, "file_write frames"),
            ("wan", expected, "file_write frames"),
        ]

    def test_video_postprocess_does_not_trust_underreported_frame_nbytes(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        class FakeFrames:
            shape = (4, 256, 256, 3)
            ndim = 4
            nbytes = 0

        frames = FakeFrames()
        expected = 4 * 256 * 256 * 3 * 4 * 6
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        captures: list[tuple[str, int, str]] = []

        for name, pipe in (("ltx", ltx), ("wan", wan)):
            pipe._check_memory = lambda phase: None  # type: ignore[method-assign]

            def capture(required_bytes: int, phase: str, *, pipe_name: str = name) -> None:
                captures.append((pipe_name, required_bytes, phase))

            pipe._check_host_allocation = capture  # type: ignore[method-assign]

        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )

        for action in (
            lambda: ltx.encode_video(frames, fps=24),
            lambda: wan.encode_video(frames, fps=24),
            lambda: ltx.write_output(frames, tmp_path / "ltx-out", run_id="r1"),
            lambda: wan.write_output(frames, tmp_path / "wan-out", run_id="r1"),
        ):
            with pytest.raises(ModuleNotFoundError, match="before initializing MLX"):
                action()

        assert captures == [
            ("ltx", expected, "video_encode frames"),
            ("wan", expected, "video_encode frames"),
            ("ltx", expected, "file_write frames"),
            ("wan", expected, "file_write frames"),
        ]

    def test_video_postprocess_import_waits_for_mlx_runtime_guard(self, tmp_path, monkeypatch):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        for pipe in (ltx, wan):
            pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
            pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
            pipe._ensure_mlx_runtime_ready = lambda phase: (_ for _ in ()).throw(MemoryGuardError("guard first"))  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise AssertionError("video postprocess must not import before runtime guard")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(MemoryGuardError, match="guard first"):
            ltx.encode_video(frames, fps=24)
        with pytest.raises(MemoryGuardError, match="guard first"):
            wan.encode_video(frames, fps=24)
        with pytest.raises(MemoryGuardError, match="guard first"):
            ltx.write_output(frames, tmp_path / "ltx-out", run_id="r1")
        with pytest.raises(MemoryGuardError, match="guard first"):
            wan.write_output(frames, tmp_path / "wan-out", run_id="r1")

    def test_video_postprocess_dependency_check_runs_before_mlx_runtime_guard(self, tmp_path, monkeypatch):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        for pipe in (ltx, wan):
            pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
            pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
            pipe._ensure_mlx_runtime_ready = lambda phase: (_ for _ in ()).throw(  # type: ignore[method-assign]
                AssertionError("runtime guard must not run before dependency preflight")
            )
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )

        for action in (
            lambda: ltx.encode_video(frames, fps=24),
            lambda: wan.encode_video(frames, fps=24),
            lambda: ltx.write_output(frames, tmp_path / "ltx-out", run_id="r1"),
            lambda: wan.write_output(frames, tmp_path / "wan-out", run_id="r1"),
        ):
            with pytest.raises(ModuleNotFoundError, match="before initializing MLX"):
                action()

    def test_video_postprocess_import_failure_after_runtime_runs_cleanup(self, tmp_path, monkeypatch):
        import numpy as np
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))
        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise RuntimeError("postprocess import failed after runtime opened")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        ltx = LTX23MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        wan = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
            save_video=True,
        )
        for pipe in (ltx, wan):
            pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
            pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
            pipe._ensure_mlx_runtime_ready = lambda phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="postprocess import failed"):
            ltx.encode_video(frames, fps=24)
        with pytest.raises(RuntimeError, match="postprocess import failed"):
            wan.encode_video(frames, fps=24)

        assert cleanup_calls == ["cleanup", "cleanup"]

    def test_wan22_missing_model_file_blocks_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="cannot stat"):
                pipe.load_model()

        limits.assert_not_called()

    def test_adapter_dependency_check_runs_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.ltx23_mlx_adapter import LTX23MLXPipeline
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        ltx_transformer = tmp_path / "ltx" / "transformer"
        ltx_transformer.mkdir(parents=True)
        (ltx_transformer / "config.json").write_text(json.dumps({"in_channels": 128}), encoding="utf-8")
        (ltx_transformer / "model.safetensors").write_bytes(b"x")

        wan_dir = tmp_path / "wan"
        wan_dir.mkdir()
        (wan_dir / "t5_encoder.safetensors").write_bytes(b"x")
        (wan_dir / "model.safetensors").write_bytes(b"x")
        (wan_dir / "config.json").write_text("{}", encoding="utf-8")
        (wan_dir / "tokenizer").mkdir()

        monkeypatch.setattr(
            "fastgen_profiler.backends.ltx23_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: None if name == "mlx_video" else object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(ModuleNotFoundError, match="before initializing MLX"):
                LTX23MLXPipeline(
                    model_path=tmp_path / "ltx",
                    seed=1,
                    width=256,
                    height=256,
                    frames=4,
                    steps=1,
                ).load_model()
            with pytest.raises(ModuleNotFoundError, match="before initializing MLX"):
                Wan22MLXPipeline(
                    model_path=wan_dir,
                    seed=1,
                    width=256,
                    height=256,
                    frames=4,
                    steps=1,
                ).load_model()

        limits.assert_not_called()

    @pytest.mark.parametrize(
        ("factory_name", "kwargs", "message"),
        [
            ("create_ltx23_pipeline", {"frames": 258}, "frames must be no greater than 257"),
            ("create_wan22_pipeline", {"steps": 513}, "steps must be no greater than 512"),
        ],
    )
    def test_adapter_factories_reject_direct_configs_outside_safe_bounds(
        self,
        tmp_path,
        factory_name,
        kwargs,
        message,
    ):
        if factory_name == "create_ltx23_pipeline":
            from fastgen_profiler.backends.ltx23_mlx_adapter import create_ltx23_pipeline as factory
        else:
            from fastgen_profiler.backends.wan22_mlx_adapter import create_wan22_pipeline as factory

        config = {
            "model_path": tmp_path,
            "seed": 1,
            "width": 256,
            "height": 256,
            "frames": 4,
            "steps": 1,
        }
        config.update(kwargs)

        with pytest.raises(ValueError, match=message):
            factory(**config)

    def test_wan22_missing_config_blocks_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "t5_encoder.safetensors").write_bytes(b"x")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "tokenizer").mkdir()
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(FileNotFoundError, match="Wan2.2 config not found"):
                pipe.load_model()

        limits.assert_not_called()

    def test_wan22_load_model_runtime_exception_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "t5_encoder.safetensors").write_bytes(b"x")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "dual_model": False,
                    "patch_size": [1, 2, 2],
                    "vae_stride": [4, 16, 16],
                    "max_area": 0,
                    "vae_z_dim": 48,
                    "dim": 4096,
                    "ffn_dim": 16_384,
                    "num_layers": 1,
                    "num_heads": 1,
                    "text_dim": 4096,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()

        config = types.SimpleNamespace(
            dual_model=False,
            patch_size=(1, 2, 2),
            vae_stride=(4, 16, 16),
            max_area=0,
            vae_z_dim=48,
            dim=4096,
            ffn_dim=16_384,
            num_layers=1,
            num_heads=1,
            text_dim=4096,
            num_train_timesteps=1000,
            sample_shift=1.0,
            model_type="wan2.2",
        )

        class FakeEncoder:
            def parameters(self):
                return object()

        fake_mx = types.SimpleNamespace(
            eval=lambda *args: None,
            clear_cache=lambda: None,
            random=types.SimpleNamespace(seed=lambda seed: None),
        )
        scheduler_module = types.ModuleType("mlx_video.models.wan_2.scheduler")
        scheduler_module.FlowUniPCScheduler = lambda num_train_timesteps: types.SimpleNamespace(
            set_timesteps=lambda steps, shift: None
        )
        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_t5_encoder = lambda path, config: FakeEncoder()
        utils_module.load_wan_model = lambda path, config, quantization: (_ for _ in ()).throw(
            RuntimeError("wan model load failed after runtime opened")
        )

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path, local_files_only):
                return object()

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, "transformers", transformers_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr("fastgen_profiler.backends.wan22_mlx_adapter._load_config", lambda path: (config, None))
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_run_budget = lambda phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan model load failed"):
            pipe.load_model()

        assert cleanup_calls == ["cleanup"]

    def test_wan22_load_config_failure_after_runtime_runs_cleanup(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "t5_encoder.safetensors").write_bytes(b"x")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "dual_model": False,
                    "patch_size": [1, 2, 2],
                    "vae_stride": [4, 16, 16],
                    "max_area": 0,
                    "vae_z_dim": 48,
                    "dim": 4096,
                    "ffn_dim": 16_384,
                    "num_layers": 1,
                    "num_heads": 1,
                    "text_dim": 4096,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter._load_config",
            lambda path: (_ for _ in ()).throw(RuntimeError("wan config import failed after runtime opened")),
        )
        cleanup_calls: list[str] = []
        monkeypatch.setattr("fastgen_profiler.mlx_guard.mlx_cleanup", lambda: cleanup_calls.append("cleanup"))

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_run_budget = lambda phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="wan config import failed"):
            pipe.load_model()

        assert cleanup_calls == ["cleanup"]

    def test_wan22_load_model_rejects_empty_loaded_parameters(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "t5_encoder.safetensors").write_bytes(b"x")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "patch_size": [1, 2, 2],
                    "vae_stride": [4, 16, 16],
                    "max_area": 0,
                    "vae_z_dim": 48,
                    "dim": 4096,
                    "ffn_dim": 16_384,
                    "num_layers": 1,
                    "num_heads": 1,
                    "text_dim": 4096,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()

        config = types.SimpleNamespace(
            dual_model=False,
            patch_size=(1, 2, 2),
            vae_stride=(4, 16, 16),
            max_area=0,
            vae_z_dim=48,
            dim=4096,
            ffn_dim=16_384,
            num_layers=1,
            num_heads=1,
            text_dim=4096,
            num_train_timesteps=1000,
            sample_shift=1.0,
            model_type="wan2.2",
        )

        class EmptyEncoder:
            def parameters(self):
                return {}

        fake_mx = types.SimpleNamespace(
            eval=lambda *args: (_ for _ in ()).throw(AssertionError("empty parameters must be rejected first")),
            clear_cache=lambda: None,
            random=types.SimpleNamespace(seed=lambda seed: None),
        )
        scheduler_module = types.ModuleType("mlx_video.models.wan_2.scheduler")
        scheduler_module.FlowUniPCScheduler = lambda num_train_timesteps: (_ for _ in ()).throw(
            AssertionError("scheduler must not be constructed with empty parameters")
        )
        utils_module = types.ModuleType("mlx_video.models.wan_2.utils")
        utils_module.load_t5_encoder = lambda path, config: EmptyEncoder()
        utils_module.load_wan_model = lambda path, config, quantization: (_ for _ in ()).throw(
            AssertionError("wan model must not load after empty t5 parameters")
        )

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path, local_files_only):
                raise AssertionError("tokenizer must not load after empty t5 parameters")

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
        monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
        monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
        monkeypatch.setitem(sys.modules, "transformers", transformers_module)
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr("fastgen_profiler.backends.wan22_mlx_adapter._load_config", lambda path: (config, None))

        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._mlx_runtime_ready = True
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_memory = lambda phase: None  # type: ignore[method-assign]
        pipe._check_host_allocation = lambda required_bytes, phase: None  # type: ignore[method-assign]
        pipe._check_run_budget = lambda phase: None  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="t5_encoder exposed no parameters"):
            pipe.load_model()

    def test_wan22_load_model_rejects_non_positive_shape_config_before_mlx_limits(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "patch_size": [1, 0, 2],
                    "vae_stride": [4, 16, 16],
                    "max_area": 0,
                    "vae_z_dim": 48,
                    "dim": 4096,
                    "ffn_dim": 16_384,
                    "num_layers": 1,
                    "num_heads": 1,
                    "text_dim": 4096,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer").mkdir()
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match=r"patch_size\[1\]=0 must be a positive integer"):
                pipe.load_model()

        limits.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("patch_size", [1, 65_537, 2], r"patch_size\[1\]=65537 exceeds safe structural dimension"),
            ("vae_stride", [4, 65_537, 16], r"vae_stride\[1\]=65537 exceeds safe structural dimension"),
            ("vae_z_dim", 65_537, r"vae_z_dim=65537 exceeds safe structural dimension"),
            ("max_area", 4096 * 4096 + 1, r"max_area=16777217 exceeds safe structural dimension"),
        ],
    )
    def test_wan22_load_model_rejects_oversized_shape_config_before_mlx_limits(
        self,
        tmp_path,
        monkeypatch,
        field,
        value,
        message,
    ):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

        config = {
            "patch_size": [1, 2, 2],
            "vae_stride": [4, 16, 16],
            "max_area": 0,
            "vae_z_dim": 48,
            "dim": 4096,
            "ffn_dim": 16_384,
            "num_layers": 1,
            "num_heads": 1,
            "text_dim": 4096,
        }
        config[field] = value
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (tmp_path / "tokenizer").mkdir()
        pipe = Wan22MLXPipeline(
            model_path=tmp_path,
            seed=1,
            width=256,
            height=256,
            frames=4,
            steps=1,
        )
        pipe._check_file_load = lambda path, phase: None  # type: ignore[method-assign]
        pipe._check_tokenizer_load = lambda path, phase: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "fastgen_profiler.backends.wan22_mlx_adapter.importlib.util.find_spec",
            lambda name: object(),
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match=message):
                pipe.load_model()

        limits.assert_not_called()

    def test_wan22_load_config_missing_file_does_not_import_mlx_video(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends.wan22_mlx_adapter import _load_config

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise AssertionError("missing Wan config must be rejected before mlx_video imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(FileNotFoundError, match="Wan2.2 config not found"):
            _load_config(tmp_path)

    def test_wan22_load_config_rejects_oversized_json_before_mlx_video(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import wan22_mlx_adapter
        from fastgen_profiler.backends.wan22_mlx_adapter import _load_config

        (tmp_path / "config.json").write_text(
            " " * (wan22_mlx_adapter._MAX_CONFIG_JSON_BYTES + 1),
            encoding="utf-8",
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("mlx_video"):
                raise AssertionError("oversized Wan config must be rejected before mlx_video imports")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeMemoryAbort, match="above safe config limit"):
            _load_config(tmp_path)

    def test_wan22_raw_config_rejects_oversized_json_before_read(self, tmp_path):
        from fastgen_profiler.backends import wan22_mlx_adapter
        from fastgen_profiler.backends.wan22_mlx_adapter import _load_raw_config_for_preflight

        config_path = tmp_path / "config.json"
        config_path.write_text(
            " " * (wan22_mlx_adapter._MAX_CONFIG_JSON_BYTES + 1),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeMemoryAbort, match="above safe config limit"):
            _load_raw_config_for_preflight(config_path)

    def test_wan22_raw_config_rejects_too_many_json_items_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends import wan22_mlx_adapter
        from fastgen_profiler.backends.wan22_mlx_adapter import _load_raw_config_for_preflight

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {f"key_{index}": index for index in range(wan22_mlx_adapter._MAX_CONFIG_JSON_ITEMS + 1)}
            ),
            encoding="utf-8",
        )

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="safe item limit"):
                _load_raw_config_for_preflight(config_path)

        limits.assert_not_called()

    def test_wan22_raw_config_rejects_deep_json_before_mlx_limits(self, tmp_path):
        from fastgen_profiler.backends import wan22_mlx_adapter
        from fastgen_profiler.backends.wan22_mlx_adapter import _load_raw_config_for_preflight

        value: object = 0
        for _ in range(wan22_mlx_adapter._MAX_CONFIG_JSON_DEPTH + 1):
            value = {"nested": value}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(value), encoding="utf-8")

        with patch("fastgen_profiler.mlx_guard.configure_mlx_resource_limits") as limits:
            with pytest.raises(RuntimeMemoryAbort, match="safe depth"):
                _load_raw_config_for_preflight(config_path)

        limits.assert_not_called()

    def test_ltx23_text_encoder_helper_is_local_first_by_default(self, tmp_path):
        from fastgen_profiler.backends.ltx23_text_encoder_download import ensure_text_encoder

        with pytest.raises(FileNotFoundError, match="Set auto_download=True explicitly"):
            ensure_text_encoder(tmp_path / "missing")

    def test_ltx23_text_encoder_helper_rejects_excessive_local_asset_scan(self, tmp_path, monkeypatch):
        from fastgen_profiler.backends import ltx23_text_encoder_download
        from fastgen_profiler.backends.ltx23_text_encoder_download import ensure_text_encoder

        text_encoder = tmp_path / "text_encoder"
        tokenizer = tmp_path / "tokenizer"
        text_encoder.mkdir()
        tokenizer.mkdir()
        (text_encoder / "config.json").write_text("{}", encoding="utf-8")
        (text_encoder / "a.safetensors").write_bytes(b"x")
        (text_encoder / "b.safetensors").write_bytes(b"x")
        (tokenizer / "tokenizer.json").write_bytes(b"x")
        (tokenizer / "tokenizer_config.json").write_bytes(b"x")
        monkeypatch.setattr(ltx23_text_encoder_download, "_MAX_TEXT_ENCODER_SCAN_FILES", 0)

        with pytest.raises(MemoryGuardError, match="exceeded 0 files"):
            ensure_text_encoder(tmp_path)
