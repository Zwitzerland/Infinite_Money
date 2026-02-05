"""CLI for interacting with QuantConnect Cloud endpoints.

This CLI uses the QuantConnect REST API via the typed connector. It requires
the QuantConnect API token to be provided via environment variables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from control_plane.connectors.quantconnect import (
    BacktestRequest,
    CompilationRequest,
    FileCreateRequest,
    QuantConnectClient,
)
from hedge_fund.utils.settings import PlatformSettings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantConnect CLI")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override QuantConnect API base URL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Create compile job")
    compile_parser.add_argument("--project-id", type=int, required=True)
    compile_parser.add_argument("--name", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Create backtest")
    backtest_parser.add_argument("--project-id", type=int, required=True)
    backtest_parser.add_argument("--name", required=True)
    backtest_parser.add_argument(
        "--parameters-json",
        default=None,
        help="JSON string of parameters.",
    )
    backtest_parser.add_argument(
        "--parameters-file",
        default=None,
        help="Path to JSON file of parameters.",
    )

    file_parser = subparsers.add_parser("upload-file", help="Create/update file")
    file_parser.add_argument("--project-id", type=int, required=True)
    file_parser.add_argument("--remote-name", required=True)
    file_parser.add_argument("--path", required=True)

    return parser.parse_args()


def _load_parameters(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.parameters_json and args.parameters_file:
        raise ValueError("Provide only one of --parameters-json or --parameters-file")
    if args.parameters_json:
        return json.loads(args.parameters_json)
    if args.parameters_file:
        return json.loads(Path(args.parameters_file).read_text())
    return {}


def _client(base_url_override: str | None) -> QuantConnectClient:
    settings = PlatformSettings()
    token = settings.quantconnect_api_token
    if token is None:
        raise ValueError(
            "QuantConnect API token missing; set INFINITE_MONEY_QUANTCONNECT_API_TOKEN."
        )
    base_url = base_url_override or settings.quantconnect_base_url
    return QuantConnectClient(base_url, token.get_secret_value())


def _print_response(response: Mapping[str, Any]) -> None:
    print(json.dumps(response, indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    try:
        client = _client(args.base_url)
        if args.command == "compile":
            payload = CompilationRequest(project_id=args.project_id, name=args.name)
            _print_response(client.create_compile_job(payload))
            return
        if args.command == "backtest":
            parameters = _load_parameters(args)
            payload = BacktestRequest(
                project_id=args.project_id,
                name=args.name,
                parameters=parameters,
            )
            _print_response(client.create_backtest(payload))
            return
        if args.command == "upload-file":
            content = Path(args.path).read_text(encoding="utf-8")
            payload = FileCreateRequest(
                project_id=args.project_id,
                name=args.remote_name,
                content=content,
            )
            _print_response(client.create_file(payload))
            return
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
