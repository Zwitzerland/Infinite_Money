from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "artifacts" / "backtest_equity.csv"
    png_path = root / "artifacts" / "backtest_equity.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing backtest equity file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["G2MAX_X"], label="G2MAX_X")
    plt.plot(df["date"], df["BuyHold"], label="BuyHold")
    plt.yscale("log")
    plt.xticks(rotation=30)
    plt.title("Backtest Equity (log scale)")
    plt.legend()
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    print(png_path)


if __name__ == "__main__":
    main()
