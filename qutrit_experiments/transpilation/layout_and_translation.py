"""Transpilation with layout and translation."""
from collections.abc import Sequence
from typing import Union
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager, TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
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


class UndoLayout(TransformationPass):
    """Undo the layout to physical qubits.

    TODO This is probably doable without the original_qubits input (remove the register named
    ancilla and use original_qubit_indices in the property set)
    """
    def __init__(self, original_qubits: Sequence[int]):
        super().__init__()
        self.original_qubits = tuple(original_qubits)

    def run(self, dag: DAGCircuit):
        q = QuantumRegister(len(self.original_qubits), 'q')
        physical_to_virtual = {pq: q[iq] for iq, pq in enumerate(self.original_qubits)}

        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        new_dag.metadata = dag.metadata
        new_dag.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        new_dag._global_phase = dag._global_phase

        for node in dag.topological_op_nodes():
            try:
                qargs = [physical_to_virtual[dag.find_bit(q).index] for q in node.qargs]
            except KeyError:
                raise TranspilerError('Op node on unused qubit found')
            new_dag.apply_operation_back(node.op, qargs, node.cargs, check=False)

        self.property_set.pop('layout')
        self.property_set.pop('original_qubit_indices')

        return new_dag


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


class TranslateAndClearTiming(BasisTranslator):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        dag = super().run(dag)
        try:
            self.property_set.pop('node_start_time')
        except KeyError:
            pass
        return dag
