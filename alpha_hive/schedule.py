import argparse, pytz, time, subprocess
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

def job_retrain(cfg_path):
    print("[SCHED] retrain start", datetime.utcnow())
    subprocess.run(["python", "-m", "alpha_hive.rl_forge", "--config", cfg_path, "--timesteps", "50000"], check=True)
    subprocess.run(["python", "-m", "alpha_hive.simulator", "--config", cfg_path, "--model_path", "artifacts/ppo_last.zip"], check=True)
    subprocess.run(["python", "-m", "alpha_hive.selector", "--stats_path", "artifacts/backtest_stats.json"], check=True)

def job_quantum():
    print("[SCHED] quantum refresh", datetime.utcnow())
    subprocess.run(["python", "-m", "alpha_hive.quantum_opt", "--csv", "data/prices.csv", "--out", "artifacts/quantum_weights.json"], check=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    tz = pytz.timezone("America/New_York")
    sch = BackgroundScheduler(timezone=tz)
    sch.add_job(job_retrain, "cron", hour=2, minute=0, args=[args.config], id="nightly_retrain")
    sch.add_job(job_quantum, "cron", hour=2, minute=30, id="quantum_refresh")
    sch.start(); print("[SCHED] running... Ctrl+C to exit")
    try: 
        while True: time.sleep(2**31-1)
    except KeyboardInterrupt:
        sch.shutdown()

if __name__ == "__main__": main()
