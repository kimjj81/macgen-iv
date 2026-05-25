"""Backend adapter registry."""

from __future__ import annotations

from .base import BackendAdapter
from .mlx import MLXBackend
from .stub import StubBackend


def create_backend(name: str) -> BackendAdapter:
    if name == "stub":
        return StubBackend()
    if name == "mlx":
        return MLXBackend()
    raise ValueError(f"Unknown backend: {name}")


__all__ = ["BackendAdapter", "create_backend"]
