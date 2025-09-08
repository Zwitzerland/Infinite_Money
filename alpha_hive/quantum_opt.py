import argparse, json, numpy as np, pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import Sampler

def mean_variance_qubo(mu, cov, lam=0.1):
    n = len(mu); qp = QuadraticProgram()
    for i in range(n): qp.binary_var(f"x_{i}")
    lin  = {f"x_{i}": float(-mu[i]) for i in range(n)}
    quad = {(f"x_{i}", f"x_{j}"): float(lam*cov[i,j]) for i in range(n) for j in range(n)}
    qp.minimize(linear=lin, quadratic=quad)
    return qp

def solve_qaoa(qp: QuadraticProgram, reps=2):
    qaoa = QAOA(sampler=Sampler(), reps=reps)
    optimizer = MinimumEigenOptimizer(qaoa)
    res = optimizer.solve(qp)
    x = np.array([res.variables_dict[f"x_{i}"] for i in range(len(res.variables))], dtype=float)
    w = x / (x.sum() + 1e-9)
    return w.tolist()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
    df = pd.read_csv(args.csv, parse_dates=["date"])
    pivot = df.pivot(index="date", columns="symbol", values="close").ffill().dropna()
    rets = pivot.pct_change().dropna(); mu = rets.mean().values; cov = rets.cov().values
    qp = mean_variance_qubo(mu, cov, lam=0.1); weights = solve_qaoa(qp, reps=2)
    out = {sym: float(w) for sym, w in zip(pivot.columns.tolist(), weights)}
    with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__": main()
