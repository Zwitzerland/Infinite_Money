"""imctl command line interface."""
from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_imctl_backtest
from .doctor import run_doctor
from .knowledge import render_traceability, sync_knowledge
from .optimize import run_imctl_optimize
from .report import build_report
from .checks import run_checks
from .ledger import create_run
from .qc_market import QCDailyEquityExport, export_qc_daily_equity_to_market_csv
from .signals import run_signal_export
from .sr import run_sr_report, run_sr_sweep


def _artifacts_root() -> Path:
    return Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(prog="imctl")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Validate environment and update docs")

    backtest = sub.add_parser("backtest", help="Run LEAN backtest")
    backtest.add_argument("--project", required=True)
    backtest.add_argument("--params", required=True)

    optimize = sub.add_parser("optimize", help="Run LEAN optimization study")
    optimize.add_argument("--study", required=True)
    optimize.add_argument("--project", required=True)
    optimize.add_argument("--n-trials", type=int, default=20)
    optimize.add_argument("--sampler", choices=["tpe", "cmaes"], default="tpe")
    optimize.add_argument("--search-space", default="optimizer/search_space.yaml")
    optimize.add_argument("--constraints", default=None)

    report = sub.add_parser("report", help="Build report for a run")
    report.add_argument("--run-id", required=True)

    knowledge = sub.add_parser("knowledge", help="Knowledge index operations")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_cmd", required=True)
    knowledge_sub.add_parser("sync", help="Extract PDFs and update index")

    sub.add_parser("checks", help="Run lint/unit/smoke checks")

    data = sub.add_parser("data", help="Dataset utilities")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    qc = data_sub.add_parser("qc-market", help="Export QC daily equity zip to market_data.csv")
    qc.add_argument("--zip", dest="zip_path", required=True)
    qc.add_argument("--symbol", required=True)
    qc.add_argument("--out", dest="output_csv", default="data/market_data.csv")
    qc.add_argument("--start", default=None)
    qc.add_argument("--end", default=None)
    qc.add_argument("--price-scale", type=float, default=10000.0)

    signals = sub.add_parser("signals", help="Signal export utilities")
    signals_sub = signals.add_subparsers(dest="signals_cmd", required=True)
    export = signals_sub.add_parser("export", help="Export AI signals to CSV")
    export.add_argument("--config", default="hedge_fund/conf/ai_stack.yaml")
    export.add_argument("--output-root", default=".")

    sr = sub.add_parser("sr", help="Support/Resistance research workflows")
    sr_sub = sr.add_subparsers(dest="sr_cmd", required=True)

    sr_report = sr_sub.add_parser("report", help="Run SR diagnostics report")
    sr_report.add_argument("--config", default="hedge_fund/conf/ai_stack_csv_sr_barrier_demo.yaml")
    sr_report.add_argument("--cost-per-turnover", type=float, default=0.0)
    sr_report.add_argument("--n-splits", type=int, default=6)
    sr_report.add_argument("--n-test-folds", type=int, default=2)
    sr_report.add_argument("--purge", type=int, default=10)
    sr_report.add_argument("--embargo", type=int, default=10)
    sr_report.add_argument("--start", default=None, help="Evaluation window start (YYYY-MM-DD)")
    sr_report.add_argument("--end", default=None, help="Evaluation window end (YYYY-MM-DD)")

    sr_sweep = sr_sub.add_parser("sweep", help="Optuna sweep over SR params")
    sr_sweep.add_argument("--config", default="hedge_fund/conf/ai_stack_csv_sr_barrier_demo.yaml")
    sr_sweep.add_argument("--search-space", default="optimizer/search_space_sr_barrier.yaml")
    sr_sweep.add_argument("--n-trials", type=int, default=30)
    sr_sweep.add_argument("--sampler", choices=["tpe", "cmaes"], default="tpe")
    sr_sweep.add_argument("--cost-per-turnover", type=float, default=0.0)
    sr_sweep.add_argument("--n-splits", type=int, default=6)
    sr_sweep.add_argument("--n-test-folds", type=int, default=2)
    sr_sweep.add_argument("--purge", type=int, default=10)
    sr_sweep.add_argument("--embargo", type=int, default=10)
    sr_sweep.add_argument("--start", default=None, help="Evaluation window start (YYYY-MM-DD)")
    sr_sweep.add_argument("--end", default=None, help="Evaluation window end (YYYY-MM-DD)")

    sr_backtest = sr_sub.add_parser("backtest", help="Export SR signals then run LEAN backtest")
    sr_backtest.add_argument("--config", default="hedge_fund/conf/ai_stack_csv_sr_barrier_demo.yaml")
    sr_backtest.add_argument("--output-root", default=".")
    sr_backtest.add_argument("--project", default="lean_projects/AISignalTrader")
    sr_backtest.add_argument("--params", default="configs/lean/ai_signal_params.yaml")

    args = parser.parse_args()
    root = Path.cwd()

    if args.command == "doctor":
        status = run_doctor(root)
        print(status)
        return

    if args.command == "backtest":
        run_dir = run_imctl_backtest(
            project=args.project,
            params_path=Path(args.params),
            artifacts_root=_artifacts_root(),
        )
        print(f"Backtest artifacts: {run_dir}")
        return

    if args.command == "optimize":
        run_dir = run_imctl_optimize(
            project=args.project,
            search_space_path=Path(args.search_space),
            constraints_path=Path(args.constraints) if args.constraints else None,
            study_name=args.study,
            n_trials=args.n_trials,
            sampler_name=args.sampler,
            artifacts_root=_artifacts_root(),
        )
        print(f"Optimization artifacts: {run_dir}")
        return

    if args.command == "report":
        run_id = args.run_id
        if run_id == "latest":
            latest_path = _artifacts_root() / "latest_run.txt"
            run_id = latest_path.read_text().strip()
        report_path = build_report(_artifacts_root() / run_id)
        print(f"Report written: {report_path}")
        return

    if args.command == "knowledge":
        sync_knowledge(root)
        traceability = render_traceability(root)
        (root / "docs" / "TRACEABILITY.md").write_text(traceability)
        print("Knowledge index updated")
        return

    if args.command == "checks":
        run = create_run(_artifacts_root())
        output_path = run.root / "checks.json"
        run_checks(root, output_path)
        print(f"Checks written: {output_path}")
        return

    if args.command == "data":
        if args.data_cmd == "qc-market":
            output_path = export_qc_daily_equity_to_market_csv(
                QCDailyEquityExport(
                    zip_path=Path(args.zip_path),
                    symbol=str(args.symbol),
                    output_csv=Path(args.output_csv),
                    price_scale=float(args.price_scale),
                    start=str(args.start) if args.start else None,
                    end=str(args.end) if args.end else None,
                )
            )
            print(f"Wrote: {output_path}")
            return

    if args.command == "signals":
        if args.signals_cmd == "export":
            output_path = run_signal_export(Path(args.config), Path(args.output_root))
            print(f"Signals written: {output_path}")
            return

    if args.command == "sr":
        if args.sr_cmd == "report":
            run_dir = run_sr_report(
                config_path=Path(args.config),
                artifacts_root=_artifacts_root(),
                cost_per_turnover=float(args.cost_per_turnover),
                n_splits=int(args.n_splits),
                n_test_folds=int(args.n_test_folds),
                purge=int(args.purge),
                embargo=int(args.embargo),
                start=str(args.start) if args.start else None,
                end=str(args.end) if args.end else None,
            )
            print(f"SR report artifacts: {run_dir}")
            return
        if args.sr_cmd == "sweep":
            run_dir = run_sr_sweep(
                config_path=Path(args.config),
                search_space_path=Path(args.search_space),
                artifacts_root=_artifacts_root(),
                n_trials=int(args.n_trials),
                sampler_name=str(args.sampler),
                cost_per_turnover=float(args.cost_per_turnover),
                n_splits=int(args.n_splits),
                n_test_folds=int(args.n_test_folds),
                purge=int(args.purge),
                embargo=int(args.embargo),
                start=str(args.start) if args.start else None,
                end=str(args.end) if args.end else None,
            )
            print(f"SR sweep artifacts: {run_dir}")
            return
        if args.sr_cmd == "backtest":
            output_path = run_signal_export(Path(args.config), Path(args.output_root))
            run_dir = run_imctl_backtest(
                project=str(args.project),
                params_path=Path(args.params),
                artifacts_root=_artifacts_root(),
            )
            print(f"Signals: {output_path}")
            print(f"Backtest artifacts: {run_dir}")
            return
