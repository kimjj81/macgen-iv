"""Profiler orchestration."""

from __future__ import annotations

from .backends.base import BackendAdapter
from .metrics import MeasurementRecord, RunConfig, machine_metadata, new_run_id, utc_timestamp


class Profiler:
    """Coordinates one profiler run through a backend adapter."""

    def __init__(self, backend: BackendAdapter) -> None:
        self.backend = backend

    def run(self, config: RunConfig) -> list[MeasurementRecord]:
        return self.backend.run(
            config,
            run_id=new_run_id(),
            timestamp_utc=utc_timestamp(),
            machine=machine_metadata(),
        )
