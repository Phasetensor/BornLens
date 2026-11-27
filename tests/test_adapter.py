import importlib.util
import unittest

from bornlens import Sample, compare


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit dependency")
class AdapterTests(unittest.TestCase):
    def test_axes_and_order(self):
        from qiskit import QuantumCircuit

        from bornlens.qiskit import from_counts, sample_counts

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.h(1)
        circuit.s(1)
        for label, sign in [("YZ", -1), ("YI", 1), ("IZ", -1)]:
            counts = sample_counts(circuit, label, 60, seed=31)
            self.assertEqual(from_counts(counts, label).mean, sign)
        x = QuantumCircuit(1)
        x.h(0)
        self.assertEqual(sample_counts(x, "X", 60, seed=4), {"0": 60})

    def test_same_counts(self):
        from unittest.mock import patch

        from qiskit import QuantumCircuit

        from bornlens.qiskit import compare_circuits, sample_counts

        a = QuantumCircuit(1)
        a.h(0)
        b = QuantumCircuit(1)
        b.ry(0.4, 0)
        observed = []

        def record(*args, **kwargs):
            counts = sample_counts(*args, **kwargs)
            observed.append(Sample("Z", counts.get("0", 0), counts.get("1", 0)))
            return counts

        with patch("bornlens.qiskit.sample_counts", side_effect=record):
            actual = compare_circuits(
                a, b, "Z", shots_a=900, shots_b=1600, tolerance=0.1, seed=19
            )
        self.assertEqual(actual, compare(*observed, tolerance=0.1))
        self.assertEqual(actual.verdict, "beyond_tolerance")

    def test_invalid_inputs(self):
        from qiskit import QuantumCircuit

        from bornlens.qiskit import from_counts, sample_counts

        for counts in [{}, {"01": 4}, {"0 1": 3}, {"1": -1}]:
            with self.assertRaises(ValueError):
                from_counts(counts, "Z")
        circuit = QuantumCircuit(1)
        circuit.measure_all()
        with self.assertRaises(ValueError):
            sample_counts(circuit, "Z", 100)
