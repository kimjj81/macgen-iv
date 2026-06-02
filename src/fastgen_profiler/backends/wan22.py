"""Wan2.2 MLX backend placeholder."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name != "Wan22Backend":
        raise AttributeError(name)
    from .mlx import MLXBackend

    class Wan22Backend(MLXBackend):
        name = "wan2.2"

    return Wan22Backend


__all__ = ["Wan22Backend"]
