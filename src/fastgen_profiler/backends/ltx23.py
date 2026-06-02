"""LTX2.3 MLX backend placeholder."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name != "LTX23Backend":
        raise AttributeError(name)
    from .mlx import MLXBackend

    class LTX23Backend(MLXBackend):
        name = "ltx2.3"

    return LTX23Backend


__all__ = ["LTX23Backend"]
