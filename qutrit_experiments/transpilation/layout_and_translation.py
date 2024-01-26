"""Transpilation with layout and translation."""
from collections.abc import Sequence
from typing import Union
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SetLayout
)


def generate_layout_passmanager(
    physical_qubits: Sequence[int],
    coupling_map: CouplingMap
) -> PassManager:
    """A trivial layout pass manager."""
    return PassManager(
        [
            SetLayout(list(physical_qubits)),
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )


def map_to_physical_qubits(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    physical_qubits: Sequence[int],
    coupling_map: CouplingMap,
    common_circuit_optimization=False
) -> QuantumCircuit:
    """Run a pass manager with layout stage only. Assumes all input circuits have the same qreg
    structure.

    Set common_circuit_optimization=True for special-case shortcut where the only differences among
    the circuits are in calibrations and metadata.
    """
    pass_manager = StagedPassManager(
        ["layout"],
        layout=generate_layout_passmanager(physical_qubits, coupling_map)
    )

    if common_circuit_optimization and isinstance(circuit, list):
        first_circuit = pass_manager.run(circuit[0])
        transpiled_circuits = [first_circuit]
        for original in circuit[1:]:
            tcirc = first_circuit.copy()
            tcirc.calibrations = original.calibrations
            tcirc.metadata = original.metadata
            transpiled_circuits.append(tcirc)
        return transpiled_circuits

    return pass_manager.run(circuit)


def generate_translation_passmanager(
    operation_names: list[str]
) -> PassManager:
    """Generate a trivial translation passmanager."""
    return PassManager(
        [
            BasisTranslator(sel, operation_names)
        ]
    )


def translate_to_basis(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    backend: Backend
) -> QuantumCircuit:
    """Run a pass manager with layout and translation stages only."""
    return StagedPassManager(
        ["translation"],
        translation=generate_translation_passmanager(backend.operation_names)
    ).run(circuit)


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
