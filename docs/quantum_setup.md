# Q1: Local Quantum-Computing Setup

This repository's quantum-computing experiments use Qiskit with the local Aer
simulator. Q1 only validates the environment; it does not create split
manifests, train quantum models, or change the Phase 1 classical pipeline.

## Install

Use the repository's existing virtual environment convention:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. The quantum dependencies are bounded in
`requirements.txt` for the Qiskit Machine Learning 0.9.x V2-primitives stack:

- `qiskit>=2.0,<3`
- `qiskit-aer>=0.17,<0.18`
- `qiskit-machine-learning>=0.9,<0.10`

These bounds assume the official Qiskit Machine Learning 0.9.x APIs and the
Qiskit 2.x V2-primitives interface. If dependency resolution reports a
platform-specific incompatibility, recreate `.venv` with Python 3.10 or newer
before changing version bounds.

## Smoke test

Run the focused environment check with:

```bash
python -m pytest -q tests/test_quantum_environment.py
```

The test imports Qiskit, Qiskit Aer, and Qiskit Machine Learning, builds a
one-qubit Hadamard circuit, and executes it with `AerSimulator` using a fixed
seed. It does not access the BISINDO dataset or write experiment artifacts.

If the quantum packages are not installed, the test is reported as skipped by
`pytest.importorskip`; install `requirements.txt` to exercise the full check.

## Hardware policy

Local simulator tests do not require an IBM Quantum account, API token, cloud
connection, or hardware credentials. Hardware execution is opt-in and is not
part of normal installation or test commands. Future hardware tickets must
select a backend explicitly, load credentials through the user's environment
or Qiskit's supported account mechanism, and record backend metadata. No
credentials belong in this repository.

## Expected imports

The Q1 environment must provide these imports:

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit_machine_learning
```

The Q0 protocol's ideal local simulator remains the default for later quantum
experiments. No hardware backend is selected by this setup document.
