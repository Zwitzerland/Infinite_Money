"""Execute latest signal on IBKR paper account.

This is paper-only and uses a dry-run by default. Use --execute with a
confirmation flag to submit orders.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datetime import datetime

import pandas as pd
from ib_insync import IB, MarketOrder, Stock


@dataclass(frozen=True)
class ExecConfig:
    host: str
    port: int
    client_id: int
    account: Optional[str]
    csv_path: Path
    symbol: Optional[str]
    signal_mode: str
    threshold: float
    max_exposure: float
    max_order_notional: float
    max_position_notional: float
    min_cash_buffer: float
    min_hold_days: int
    log_path: Path
    approve: bool
    execute: bool
    confirm: bool


def _parse_args() -> ExecConfig:
    parser = argparse.ArgumentParser(description="Execute latest AI signal on IBKR paper")
    parser.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("IBKR_PORT", "7497")))
    parser.add_argument("--client-id", type=int, default=int(os.getenv("IBKR_CLIENT_ID", "7")))
    parser.add_argument("--account", default=os.getenv("IBKR_ACCOUNT"))
    parser.add_argument("--csv", default="data/custom/ai_signals.csv")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--signal-mode", default="exposure", choices=["exposure", "directional"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--max-order-notional", type=float, default=100000.0)
    parser.add_argument("--max-position-notional", type=float, default=200000.0)
    parser.add_argument("--min-cash-buffer", type=float, default=500.0)
    parser.add_argument("--min-hold-days", type=int, default=2)
    parser.add_argument("--log-path", default="artifacts/pdt_trades.csv")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Prompt for interactive approval before order submission.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --execute (paper only).",
    )
    args = parser.parse_args()
    return ExecConfig(
        host=str(args.host),
        port=int(args.port),
        client_id=int(args.client_id),
        account=str(args.account) if args.account else None,
        csv_path=Path(args.csv),
        symbol=str(args.symbol) if args.symbol else None,
        signal_mode=str(args.signal_mode),
        threshold=float(args.threshold),
        max_exposure=float(args.max_exposure),
        max_order_notional=float(args.max_order_notional),
        max_position_notional=float(args.max_position_notional),
        min_cash_buffer=float(args.min_cash_buffer),
        min_hold_days=int(args.min_hold_days),
        log_path=Path(args.log_path),
        approve=bool(args.approve),
        execute=bool(args.execute),
        confirm=bool(args.confirm),
    )


def _latest_signal(csv_path: Path, symbol: Optional[str]) -> tuple[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Signal file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Signal file is empty")
    if symbol:
        df = df[df["symbol"] == symbol]
    if df.empty:
        raise ValueError("No rows for requested symbol")
    row = df.iloc[-1]
    return str(row["symbol"]), float(row["signal"])


def _account_summary(ib: IB, account: Optional[str]) -> dict[str, str]:
    summary = ib.accountSummary(account=account) if account else ib.accountSummary()
    return {item.tag: item.value for item in summary}  # type: ignore[attr-defined]


def _current_position(ib: IB, symbol: str) -> float:
    for pos in ib.positions():
        if pos.contract.symbol == symbol:
            return float(pos.position)
    return 0.0


def _load_trade_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price", "position_after"])
    return pd.read_csv(path)


def _append_trade_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _load_trade_log(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def _last_entry_date(log: pd.DataFrame, symbol: str) -> Optional[datetime]:
    if log.empty:
        return None
    df = log[log["symbol"] == symbol]
    if df.empty:
        return None
    df = df.sort_values("timestamp")
    last_entry = None
    for _, row in df.iterrows():
        try:
            position_after = float(row.get("position_after", 0))
        except Exception:
            position_after = 0.0
        if position_after != 0:
            try:
                last_entry = datetime.fromisoformat(str(row["timestamp"]))
            except Exception:
                last_entry = None
    return last_entry


def _latest_price(ib: IB, symbol: str) -> float:
    contract = Stock(symbol, "SMART", "USD")
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    price = ticker.marketPrice() or ticker.last or ticker.close
    ib.cancelMktData(contract)
    if price:
        return float(price)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
    )
    if bars:
        return float(bars[-1].close)
    raise RuntimeError("Unable to fetch a price for symbol")


def _target_exposure(signal: float, mode: str, threshold: float, max_exposure: float) -> float:
    if abs(signal) < threshold:
        return 0.0
    if mode == "directional":
        return max_exposure if signal > 0 else -max_exposure
    return max(-max_exposure, min(max_exposure, signal))


def main() -> None:
    cfg = _parse_args()
    if cfg.port != 7497:
        raise SystemExit("Only paper trading port 7497 is supported.")
    if cfg.execute and not cfg.confirm:
        raise SystemExit("Refusing to execute without --confirm (paper only).")

    symbol, signal = _latest_signal(cfg.csv_path, cfg.symbol)

    ib = IB()
    if not ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=5):
        raise SystemExit("Unable to connect to IBKR. Ensure TWS/IB Gateway is running.")

    summary = _account_summary(ib, cfg.account)
    net_liq = float(summary.get("NetLiquidation", 0.0))
    cash = float(summary.get("AvailableFunds", net_liq))
    price = _latest_price(ib, symbol)

    target_exposure = _target_exposure(signal, cfg.signal_mode, cfg.threshold, cfg.max_exposure)
    target_notional = target_exposure * net_liq
    target_notional = max(-cfg.max_position_notional, min(cfg.max_position_notional, target_notional))
    target_qty = int(target_notional / price) if price > 0 else 0

    current_qty = _current_position(ib, symbol)
    order_qty = int(target_qty - current_qty)
    order_notional = abs(order_qty * price)

    print(f"Signal: {signal:.4f} | Mode: {cfg.signal_mode} | Target exposure: {target_exposure:.2f}")
    print(f"Net liquidation: {net_liq:.2f} | Cash: {cash:.2f} | Price: {price:.2f}")
    print(f"Current qty: {current_qty} | Target qty: {target_qty} | Order qty: {order_qty}")

    if order_qty == 0:
        print("No order required.")
        ib.disconnect()
        return
    if order_notional > cfg.max_order_notional:
        raise SystemExit("Order notional exceeds max_order_notional.")
    if cash < cfg.min_cash_buffer and order_qty > 0:
        raise SystemExit("Insufficient cash buffer for buy order.")

    log = _load_trade_log(cfg.log_path)
    last_entry = _last_entry_date(log, symbol)
    now = datetime.now()
    if last_entry is not None:
        hold_days = (now.date() - last_entry.date()).days
    else:
        hold_days = None

    reducing_position = (current_qty != 0) and (abs(target_qty) < abs(current_qty))
    if reducing_position and hold_days is not None and hold_days < cfg.min_hold_days:
        raise SystemExit(
            f"PDT guard: hold_days={hold_days} < min_hold_days={cfg.min_hold_days}."
        )

    if not cfg.execute:
        print("Dry run only. Re-run with --execute --confirm to place a paper order.")
        ib.disconnect()
        return

    if cfg.approve:
        answer = input(
            "Approve paper order? Type 'yes' to continue: "
        ).strip().lower()
        if answer != "yes":
            print("Order cancelled by user.")
            ib.disconnect()
            return

    order = MarketOrder("BUY" if order_qty > 0 else "SELL", abs(order_qty))
    trade = ib.placeOrder(Stock(symbol, "SMART", "USD"), order)
    ib.sleep(2)
    print(f"Submitted paper order: {trade.orderStatus.status}")
    position_after = current_qty + order_qty
    _append_trade_log(
        cfg.log_path,
        {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": "BUY" if order_qty > 0 else "SELL",
            "qty": abs(order_qty),
            "price": price,
            "position_after": position_after,
        },
    )
    ib.disconnect()


if __name__ == "__main__":
    main()
