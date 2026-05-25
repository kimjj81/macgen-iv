"""Backend adapter registry."""

from __future__ import annotations

from .base import BackendAdapter
from .ltx23 import LTX23Backend
from .wan22 import Wan22Backend


def create_backend(name: str, *, dry_run: bool = False) -> BackendAdapter:
    if name == "wan2.2":
        return Wan22Backend(dry_run=dry_run)
    if name == "ltx2.3":
        return LTX23Backend(dry_run=dry_run)
    raise ValueError(f"Unknown backend: {name}")


__all__ = ["BackendAdapter", "create_backend"]
