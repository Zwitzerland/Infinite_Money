"""CLI for AWS pipeline integration."""
from __future__ import annotations

import argparse

from hedge_fund.ai.aws.pipeline import (
    launch_step_functions,
    load_pipeline_config,
    run_local_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS pipeline CLI")
    parser.add_argument(
        "--config",
        default="hedge_fund/conf/aws_pipeline.yaml",
        help="Path to AWS pipeline config.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "step-functions"],
        default="local",
        help="Execution mode.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    if args.mode == "local":
        output_dir = run_local_pipeline(cfg)
        print(f"Local pipeline complete: {output_dir}")
        return

    execution_arn = launch_step_functions(cfg)
    print(f"Step Functions execution started: {execution_arn}")


if __name__ == "__main__":
    main()
