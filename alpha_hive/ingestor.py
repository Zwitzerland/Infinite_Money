import pandas as pd

def load_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df

def tqi_pull_stub() -> dict:
    # placeholder for nightly quantum intel ingestion (TQI or similar)
    return {
        "qpu_candidates": ["trapped-ion", "neutral-atom"],
        "signals": ["compiler_release", "gate_fidelity_update"],
        "vendors": ["VendorA", "VendorB"]
    }
