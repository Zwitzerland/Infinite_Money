"""Quantum feature map scaffolding."""
from __future__ import annotations

from typing import Iterable


def z_feature_map(features: Iterable[float]) -> object:
    """Build a simple Z feature map circuit using PennyLane if available."""
    try:
        import pennylane as qml
    except ImportError as exc:
        raise RuntimeError(
            "PennyLane not installed. Install with `pip install -e .[quantum]`."
        ) from exc

    wires = list(range(len(list(features))))
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(x):
        for idx, value in enumerate(x):
            qml.Hadamard(wires=idx)
            qml.RZ(value, wires=idx)
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    return circuit
