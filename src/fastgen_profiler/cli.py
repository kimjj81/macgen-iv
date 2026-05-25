"""Command line interface for fastgen-profiler."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backends import create_backend
from .metrics import BenchmarkConfig, write_jsonl
from .profiler import Profiler
from .reports.markdown import render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastgen-profiler",
        description="Profile MLX video generation pipelines and write benchmark JSONL.",
    )
    parser.add_argument("--model", choices=["wan2.2", "ltx2.3"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--precision", default="unknown")
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and emit a placeholder record without loading a model.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        model=args.model,
        backend=args.model,
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        precision=args.precision,
        guidance=args.guidance,
        cache_enabled=args.cache,
        compile_enabled=args.compile,
    )

    backend = create_backend(args.model, dry_run=args.dry_run)
    result = Profiler(backend).run(config)
    write_jsonl(args.jsonl, result)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown_report(result), encoding="utf-8")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
