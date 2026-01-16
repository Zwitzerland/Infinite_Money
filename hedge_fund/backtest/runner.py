"""Backtest runner entry point.

Notes
-----
This is a minimal stub for a standardized entry point. Wire it to the
backtesting engine once the event-driven pipeline is implemented.
"""
from __future__ import annotations

import sys


def main() -> None:
    """Run a backtest.

    Raises
    ------
    NotImplementedError
        Until the backtest harness is wired.
    """
    raise NotImplementedError(
        "TODO(user): Implement backtest runner with CPCV + embargo support."
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        raise
