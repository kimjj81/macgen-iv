"""Tests for mlx_guard: system snapshot, adaptive batch, runtime watchdog."""

from __future__ import annotations

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
    configure_mlx_resource_limits,
    estimate_video_run_floor_bytes,
    adaptive_batch_config_from_run,
)


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


class TestConfigureMlxResourceLimits:
    def test_sets_conservative_default_limits(self, monkeypatch):
        import mlx.core as mx

        calls: dict[str, int] = {}
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

        assert calls["memory"] == int(128 * 1024 ** 3 * 0.80)
        assert calls["cache"] == 1 * 1024 ** 3
        assert calls["wired"] == calls["memory"]
        assert status["memory_limit_gb"] is not None

    def test_env_overrides_limits(self, monkeypatch):
        import mlx.core as mx

        calls: dict[str, int] = {}
        monkeypatch.setenv("FASTGEN_MLX_MEMORY_LIMIT_GB", "12")
        monkeypatch.setenv("FASTGEN_MLX_CACHE_LIMIT_GB", "0.5")
        monkeypatch.setenv("FASTGEN_MLX_WIRED_LIMIT_GB", "10")
        monkeypatch.setattr(mx, "set_memory_limit", lambda value: calls.setdefault("memory", value) or 0)
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: calls.setdefault("cache", value) or 0)
        monkeypatch.setattr(mx, "set_wired_limit", lambda value: calls.setdefault("wired", value) or 0)

        configure_mlx_resource_limits(
            snapshot=SystemSnapshot(
                free_bytes=None,
                total_bytes=128 * 1024 ** 3,
                pressure=None,
                swap_files=None,
                free_fraction=None,
            ),
            label="limits",
        )

        assert calls["memory"] == 12 * 1024 ** 3
        assert calls["cache"] == int(0.5 * 1024 ** 3)
        assert calls["wired"] == 10 * 1024 ** 3


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

    def test_video_run_floor_scales_with_shape(self):
        small = estimate_video_run_floor_bytes(width=256, height=256, frames=4, guidance=1.0)
        large = estimate_video_run_floor_bytes(width=1024, height=1024, frames=16, guidance=3.5)
        assert large > small

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
    def test_wan22_denoise_checks_memory_before_work(self, tmp_path):
        from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline

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
        pipe.seq_len = 1
        pipe.cross_kv = object()
        pipe.rope_cos_sin = object()

        with patch(
            "fastgen_profiler.mlx_guard.check_runtime_memory",
            side_effect=RuntimeMemoryAbort("stop before wan work"),
        ):
            with pytest.raises(RuntimeMemoryAbort, match="stop before wan work"):
                pipe.denoise_step(object(), step_index=0, steps=1, guidance=1.0, cache="none")

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
