"""Smoke checks for the Q1 local quantum-computing environment."""

import pytest


def test_qiskit_local_simulator_environment():
    """Build and execute a one-qubit circuit without hardware credentials."""

    qiskit = pytest.importorskip("qiskit")
    aer = pytest.importorskip("qiskit_aer")
    qiskit_machine_learning = pytest.importorskip("qiskit_machine_learning")

    from qiskit import QuantumCircuit, transpile

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()

    simulator = aer.AerSimulator()
    compiled = transpile(circuit, simulator, optimization_level=1)
    result = simulator.run(compiled, shots=32, seed_simulator=42).result()
    counts = result.get_counts()

    assert counts
    assert set(counts).issubset({"0", "1"})
    assert hasattr(qiskit, "QuantumCircuit")
    assert hasattr(aer, "AerSimulator")
    assert qiskit_machine_learning.__name__ == "qiskit_machine_learning"
