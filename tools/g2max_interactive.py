"""Interactive G2MAX compounding chart.

Run:
  python -m tools.g2max_interactive
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from hedge_fund.ai.portfolio.g2max import G2MaxParams, g2max_equity_curve


def _generate_returns(seed: int = 7, years: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 252 * years
    regimes = [
        (0.12, 0.18, int(n * 0.30)),
        (0.02, 0.22, int(n * 0.20)),
        (-0.25, 0.45, int(n * 0.10)),
        (0.10, 0.24, n - int(n * 0.30) - int(n * 0.20) - int(n * 0.10)),
    ]
    blocks = []
    for mu, vol, length in regimes:
        mu_d = mu / 252.0
        vol_d = vol / np.sqrt(252.0)
        blocks.append(rng.normal(mu_d, vol_d, length))
    return np.concatenate(blocks)


def main() -> None:
    returns = _generate_returns()
    params = G2MaxParams()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    equity = g2max_equity_curve(returns, params)
    line, = ax.plot(equity, label="G2MAX")
    ax.set_title("G2MAX Compounding (Interactive)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_phi = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_lev = plt.axes([0.15, 0.07, 0.7, 0.03])
    phi_slider = Slider(ax_phi, "phi_base", 0.1, 0.8, valinit=params.phi_base)
    lev_slider = Slider(ax_lev, "leverage", 0.5, 4.0, valinit=params.leverage)

    def _update(_val: float) -> None:
        new_params = replace(params, phi_base=phi_slider.val, leverage=lev_slider.val)
        new_equity = g2max_equity_curve(returns, new_params)
        line.set_ydata(new_equity)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    phi_slider.on_changed(_update)
    lev_slider.on_changed(_update)

    plt.show()


if __name__ == "__main__":
    main()
