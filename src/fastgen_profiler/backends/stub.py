"""Deterministic no-weights backend for profiler verification."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastgen_profiler.metrics import MeasurementRecord, RunConfig

from .base import BackendAdapter, timed_section


class StubBackend(BackendAdapter):
    name = "stub"

    def run(
        self,
        config: RunConfig,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
    ) -> list[MeasurementRecord]:
        records: list[MeasurementRecord] = []
        total_started = perf_counter()

        output_path: str | None = None
        for phase in (
            "model_load",
            "prompt_prepare",
            "text_encoder",
            "latent_init",
        ):
            records.append(self._phase_record(config, run_id, timestamp_utc, machine, phase))

        denoise_started = perf_counter()
        for step_index in range(config.steps):
            with timed_section() as timing:
                _deterministic_work(config.seed + step_index)
            records.append(
                self.record(
                    config,
                    run_id=run_id,
                    timestamp_utc=timestamp_utc,
                    machine=machine,
                    phase="denoise_step",
                    step_index=step_index,
                    seconds=timing["seconds"],
                )
            )
        records.append(
            self.record(
                config,
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                machine=machine,
                phase="denoise_total",
                seconds=perf_counter() - denoise_started,
            )
        )

        for phase in ("vae_decode", "video_encode"):
            records.append(self._phase_record(config, run_id, timestamp_utc, machine, phase))

        with timed_section() as timing:
            if config.save_video and not config.dry_run:
                output_path = str(_write_placeholder_video(config.output_dir, run_id))
            else:
                config.output_dir.mkdir(parents=True, exist_ok=True)
        records.append(
            self.record(
                config,
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                machine=machine,
                phase="file_write",
                seconds=timing["seconds"],
                output_path=output_path,
            )
        )

        records.append(
            self.record(
                config,
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                machine=machine,
                phase="total",
                seconds=perf_counter() - total_started,
                output_path=output_path,
            )
        )
        return records

    def _phase_record(
        self,
        config: RunConfig,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
        phase: str,
    ) -> MeasurementRecord:
        with timed_section() as timing:
            _deterministic_work(config.seed + len(phase))
        return self.record(
            config,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            machine=machine,
            phase=phase,
            seconds=timing["seconds"],
        )


def _deterministic_work(seed: int) -> int:
    value = seed & 0xFFFF
    for index in range(128):
        value = (value * 1103515245 + 12345 + index) & 0x7FFFFFFF
    return value


def _write_placeholder_video(output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.stub.mp4"
    output_path.write_bytes(b"fastgen-profiler stub video placeholder\n")
    return output_path
