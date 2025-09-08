import argparse, json
from .utils import load_json

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats_path", required=True); args = ap.parse_args()
    stats = load_json(args.stats_path)
    promotable = (stats["sharpe"] > 2.0) and (stats["max_drawdown"] < 0.08)
    verdict = {"promotable": promotable, "reason": "rule_check", "stats": stats}
    print(json.dumps(verdict, indent=2))
    with open("artifacts/selection.json", "w", encoding="utf-8") as f: json.dump(verdict, f, indent=2)

if __name__ == "__main__": main()
