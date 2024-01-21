"""Transpilation with layout and translation."""
from collections.abc import Sequence
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import BasisTranslator

from .layout_only import generate_layout_passmanager


def generate_translation_passmanager(
    operation_names: list[str]
) -> PassManager:
    """Generate a trivial translation passmanager."""
    return PassManager(
        [
            BasisTranslator(sel, operation_names)
        ]
    )


def map_and_translate(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    physical_qubits: Sequence[int],
    backend: Backend
) -> QuantumCircuit:
    """Run a pass manager with layout and translation stages only."""
    return StagedPassManager(
        ["layout", "translation"],
        layout=generate_layout_passmanager(physical_qubits, backend.coupling_map),
        translation=generate_translation_passmanager(backend.operation_names)
    ).run(circuit)
