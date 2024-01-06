"""Functions to define and run layout-only PassManagers."""
from collections.abc import Sequence
from typing import Union
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import Layout, PassManager, StagedPassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import (
    ApplyLayout,
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
    coupling_map: CouplingMap
) -> QuantumCircuit:
    """Run a pass manager with layout stage only. Assumes all input circuits have the same qreg
    structure."""
    return StagedPassManager(
        ["layout"],
        layout=generate_layout_passmanager(physical_qubits, coupling_map)
    ).run(circuit)


def replace_calibration_and_metadata(
    circuits: list[QuantumCircuit],
    physical_qubits: Sequence[int],
    coupling_map: CouplingMap
):
    """Special-case shortcut where the only differences among the circuits are in calibrations and
    metadata."""
    first_circuit = map_to_physical_qubits(circuits[0], physical_qubits, coupling_map)

    transpiled_circuits = [first_circuit]
    for circuit in circuits[1:]:
        tcirc = first_circuit.copy()
        tcirc.calibrations = circuit.calibrations
        tcirc.metadata = circuit.metadata
        transpiled_circuits.append(tcirc)

    return transpiled_circuits
