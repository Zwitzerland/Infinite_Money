from g2max_x_lab import run_simulation


def test_run_simulation_deterministic() -> None:
    eq, bh = run_simulation(seed=7)
    eq2, bh2 = run_simulation(seed=7)
    assert len(eq) == len(bh) == 2520
    assert eq.iloc[-1] == eq2.iloc[-1]
    assert bh.iloc[-1] == bh2.iloc[-1]
