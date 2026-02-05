"""CLI entry point for the AI stack."""
from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI stack pipeline")
    parser.add_argument(
        "--config",
        default="hedge_fund/conf/ai_stack.yaml",
        help="Path to the AI stack config file.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/ai_runs",
        help="Root directory for run artifacts.",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config, output_root=args.output_root)


if __name__ == "__main__":
    main()
