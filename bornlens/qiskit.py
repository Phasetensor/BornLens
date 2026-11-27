"""Ideal statevector sampling for bound unitary Qiskit circuits."""

from collections.abc import Mapping

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info import Statevector

from . import Sample, _integer, compare


def from_counts(counts, observable):
    """Convert basis-rotated counts with one classical bit per qubit."""
    Sample(observable, 1, 0)
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping")
    mask = int("".join("0" if p == "I" else "1" for p in observable), 2)
    totals = [0, 0]
    for bits, count in counts.items():
        if (not isinstance(bits, str) or len(bits) != len(observable)
                or any(bit not in "01" for bit in bits)):
            raise ValueError("expected one binary digit per qubit, in a single register")
        parity = bin(int(bits, 2) & mask).count("1") % 2
        totals[parity] += _integer(count, "count")
    return Sample(observable, *totals)


def sample_counts(circuit, observable, shots, *, seed=None):
    """Sample after rotating each measured Pauli axis to the Z basis."""
    Sample(observable, 1, 0)
    shots = _integer(shots, "shots", 1)
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != len(observable):
        raise ValueError("circuit width must match the Pauli string")
    if circuit.num_parameters or any(
        not isinstance(item.operation, Gate) and item.operation.name != "barrier"
        for item in circuit.data
    ):
        raise ValueError("expected a bound gate circuit without measurements or resets")
    rotated = circuit.copy()
    for qubit, axis in enumerate(reversed(observable)):
        if axis == "Y":
            rotated.sdg(qubit)
        if axis in "XY":
            rotated.h(qubit)
    state = Statevector.from_instruction(rotated)
    state.seed(seed)
    return dict(state.sample_counts(shots))


def compare_circuits(a, b, observable, *, shots_a, shots_b, tolerance,
                     confidence=0.95, seed=None, method="hoeffding"):
    """Run both ideal circuits using separate random streams."""
    streams = np.random.SeedSequence(seed).spawn(2)
    samples = [
        from_counts(sample_counts(circuit, observable, shots,
                                  seed=np.random.default_rng(stream)), observable)
        for circuit, shots, stream in zip((a, b), (shots_a, shots_b), streams)
    ]
    return compare(*samples, tolerance=tolerance, confidence=confidence, method=method)
