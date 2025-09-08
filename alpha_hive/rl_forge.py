import argparse, os, mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from .config import load_config
from .ingestor import load_prices_csv
from .features import make_features
from .envs import SimplePortfolioEnv
from .utils import set_seed

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--timesteps", type=int, default=None); args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg.data.get("seed", 1337)); assets = cfg.data["assets"]
    df = make_features(load_prices_csv("data/prices.csv"))
    env = SimplePortfolioEnv(df, assets, cost_bps=cfg.data["train"]["transaction_cost_bps"], window=cfg.data["train"]["window"])
    check_env(env, warn=True); venv = DummyVecEnv([lambda: env])
    timesteps = args.timesteps or cfg.data["train"]["timesteps"]
    os.makedirs("artifacts", exist_ok=True)
    with mlflow.start_run(run_name="ppo-train"):
        model = PPO("MlpPolicy", venv, verbose=0, learning_rate=cfg.data["train"]["lr"],
                    gamma=cfg.data["train"]["gamma"], ent_coef=cfg.data["train"]["ent_coef"],
                    clip_range=cfg.data["train"]["clip_range"])
        model.learn(total_timesteps=timesteps); model.save("artifacts/ppo_last"); mlflow.log_param("timesteps", timesteps); mlflow.log_artifact("artifacts/ppo_last.zip")

if __name__ == "__main__": main()
