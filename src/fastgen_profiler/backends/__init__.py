"""Backend adapter registry."""

from __future__ import annotations

from .base import BackendAdapter


def create_backend(name: str) -> BackendAdapter:
    if name == "stub":
        from .stub import StubBackend

        return StubBackend()
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend()
    raise ValueError(f"Unknown backend: {name}")


__all__ = ["BackendAdapter", "create_backend"]
