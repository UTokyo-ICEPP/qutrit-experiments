"""Transpiler pass for qutrit-qubit backward generalized CX."""
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import InstructionDurations, TransformationPass, TranspilerError

from ..gates import P2Gate, QutritQubitCXType, X12Gate, XplusGate
from .dynamical_decoupling import DDCalculator


class ReverseCXDecomposition(TransformationPass):
    """Decompose QutritQubitCXGate to the double reverse ECR sequence with basis cycling.

    The circuit for CZ with basis cycling is the following:
         ┌─────┐┌─=─┐┌───────────┐┌───────────┐┌─=====─┐┌─────┐┌───┐┌───────────────┐┌─────┐┌───┐
    q_0: ┤ X12 ├┤ X ├┤1          ├┤1          ├┤ Rx(-π)├┤ X12 ├┤ X ├┤ Delay(d2[dt]) ├┤ X12 ├┤ X ├
         └─────┘└─=─┘│  Rzx(π/2) ││  Rzx(π/2) │├─=====┬┘└─────┘└───┘└───────────────┘└─────┘└───┘
    q_1: ────────────┤0          ├┤0          ├┤ Rz(π)├──────────────────────────────────────────
                     └───────────┘└───────────┘└──────┘
    where X and Rx(-π) cancel to P2(-π/2).
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        apply_dd: bool = True,
        pulse_alignment: Optional[int] = None
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.apply_dd = apply_dd
        self.pulse_alignment = pulse_alignment
        if self.apply_dd and not self.pulse_alignment:
            raise TranspilerError('Pulse alignment needed when applying DD')
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cx_cz_nodes = dag.named_nodes('qutrit_qubit_cx') + dag.named_nodes('qutrit_qubit_cz')
        for node in cx_cz_nodes:
            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids)
            if rcr_type != QutritQubitCXType.REVERSE:
                continue

            circuit = reverse2q_decomposition_circuit(node.op.name, qids, self.inst_durations,
                                                      apply_dd=self.apply_dd,
                                                      pulse_alignment=self.pulse_alignment)
            dag.substitute_node_with_dag(node, circuit_to_dag(circuit))

        return dag


def reverse2q_decomposition_circuit(
    gate: str,
    physical_qubits: tuple[int, int],
    instruction_durations: InstructionDurations,
    apply_dd: bool = True,
    pulse_alignment: int = 1,
    delta_cz: float = 0.,
    include_last_local: bool = True
) -> QuantumCircuit:
    """Return a qutrit-qubit CZ circuit with reverse ECR."""
    def dur(gate, *qubits):
        return instruction_durations.get(gate, tuple(physical_qubits[iq] for iq in qubits))

    x12_to_x12 = dur('x12', 0) + 2 * dur('ecr', 1, 0) + dur('x', 1)
    xplus_dur = dur('x12', 0) + dur('x', 0)

    circuit = QuantumCircuit(2)

    if gate == 'qutrit_qubit_cx':
        circuit.delay(dur('sx', 1), 0)
        circuit.h(1)

    if delta_cz:
        circuit.rz(-delta_cz, 0)

    circuit.append(X12Gate(), [0])
    circuit.ecr(1, 0)
    circuit.x(1)
    circuit.ecr(1, 0)
    circuit.x(1)
    circuit.append(P2Gate(-np.pi / 2.), [0])
    circuit.rz(-np.pi, 1)
    circuit.append(XplusGate(), [0])
    circuit.delay(x12_to_x12 - xplus_dur, 0)
    if include_last_local:
        circuit.append(XplusGate(), [0])

    if apply_dd:
        circuit.delay(dur('x12', 0), 1)
        ddapp = DDCalculator(physical_qubits, instruction_durations, pulse_alignment)
        ddapp.append_dd(circuit, 1, x12_to_x12 - xplus_dur, distribution='left')
    else:
        circuit.delay(x12_to_x12 - dur('x', 1), 1)

    if include_last_local:
        circuit.delay(xplus_dur, 1)

    if gate == 'qutrit_qubit_cx':
        circuit.delay(dur('sx', 1), 0)
        circuit.h(1)

    return circuit
