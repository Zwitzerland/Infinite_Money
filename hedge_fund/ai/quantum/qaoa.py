"""QAOA circuit scaffolding for portfolio QUBOs."""
from __future__ import annotations

from typing import Iterable


def build_qaoa_circuit(
    qubo: dict[tuple[int, int], float],
    beta: Iterable[float],
    gamma: Iterable[float],
) -> object:
    """Build a QAOA circuit for a QUBO (requires Braket or PennyLane).

    This is a scaffold; tune for your device and cost function.
    """
    try:
        from braket.circuits import Circuit
    except ImportError as exc:
        raise RuntimeError(
            "Braket SDK not installed. Install with `pip install -e .[quantum]`."
        ) from exc

    circuit = Circuit()
    n = max(max(i, j) for i, j in qubo.keys()) + 1
    for idx in range(n):
        circuit.h(idx)

    for layer, (b, g) in enumerate(zip(beta, gamma)):
        for (i, j), weight in qubo.items():
            if i == j:
                circuit.rz(i, 2 * g * weight)
            else:
                circuit.cnot(i, j)
                circuit.rz(j, 2 * g * weight)
                circuit.cnot(i, j)
        for idx in range(n):
            circuit.rx(idx, 2 * b)
    return circuit
