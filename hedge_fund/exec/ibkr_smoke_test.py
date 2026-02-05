"""IBKR connection smoke test using ib_insync.

This script verifies connectivity and reads basic account summary data. It does
not place orders.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from ib_insync import IB


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IBKR API smoke test")
    parser.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_int_env("IBKR_PORT", 7497))
    parser.add_argument(
        "--client-id",
        type=int,
        default=_int_env("IBKR_CLIENT_ID", 1),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_int_env("IBKR_TIMEOUT", 5),
        help="Connection timeout in seconds.",
    )
    parser.add_argument(
        "--account",
        default=os.getenv("IBKR_ACCOUNT"),
        help="Optional account id to scope account summary.",
    )
    return parser.parse_args()


def _print_summary(summary: Iterable[object]) -> None:
    try:
        summary_map = {item.tag: item.value for item in summary}  # type: ignore[attr-defined]
    except AttributeError:
        print("Account summary unavailable.")
        return
    for tag in ("NetLiquidation", "BuyingPower", "AvailableFunds", "EquityWithLoanValue"):
        value = summary_map.get(tag)
        if value is not None:
            print(f"{tag}: {value}")


def main() -> None:
    args = _parse_args()
    ib = IB()
    try:
        connected = ib.connect(
            args.host,
            args.port,
            clientId=args.client_id,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not connected:
        print("Connection failed: unknown error", file=sys.stderr)
        raise SystemExit(1)

    print(f"Connected to IBKR at {args.host}:{args.port} (client_id={args.client_id})")
    print(f"Server time: {ib.reqCurrentTime()}")

    summary = ib.accountSummary(account=args.account) if args.account else ib.accountSummary()
    _print_summary(summary)

    ib.disconnect()


if __name__ == "__main__":
    main()
