# Infinite_Money — AI Alpha Hive

Operational scaffold for a federated swarm of RL trading agents with quantum-augmented allocation.

## Quick start
```bash
bash bootstrap.sh
# or:
make setup && make all

Train → Backtest → Select

Trains PPO on toy panel, backtests, writes metrics to artifacts/.

Quantum allocation demo writes artifacts/quantum_weights.json.

Next steps

Swap data/prices.csv for Kafka/Delta feeds.

Connect IBKR in alpha_hive/executor.py (paper first).

Point MLflow to prod, wire Grafana dashboard.

No system guarantees profits. Operate under your jurisdiction’s regulations.
