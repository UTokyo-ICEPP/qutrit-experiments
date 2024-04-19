"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, BasisTranslator
from qiskit_experiments.calibration_management import Calibrations

from ..transpilation.qutrit_qubit_cx import ReverseCXDecomposition, ReverseCXDynamicalDecoupling
from ..transpilation.qutrit_toffoli import (QutritToffoliRefocusing,
                                            QutritToffoliDynamicalDecoupling)
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.layout_and_translation import (UndoLayout, TranslateAndClearTiming,
                                                    generate_layout_passmanager,
                                                    generate_translation_passmanager)
from ..gates import QutritToffoliGate


def qutrit_toffoli_circuit(
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int]
) -> QuantumCircuit:
    pm = qutrit_toffoli_translator(backend, calibrations, physical_qubits)
    circuit = QuantumCircuit(3)
    circuit.append(QutritToffoliGate(), [0, 1, 2])
    return pm.run(circuit)


def qutrit_toffoli_translator(
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int],
    do_phase_corr: bool = True,
    do_dd: bool = True
) -> StagedPassManager:
    """Return a pass manager to translate a qubit-qutrit-qubit circuit containing QutritToffoliGate or something similar."""
    physical_qubits = tuple(physical_qubits)
    x_duration = calibrations.get_schedule('x', physical_qubits[1]).duration
    x12_duration = calibrations.get_schedule('x12', physical_qubits[1]).duration

    instruction_durations = make_instruction_durations(backend, calibrations, physical_qubits)
    instruction_durations.update([
        ('x12', physical_qubits[1], x12_duration),
        ('xplus', physical_qubits[1], x_duration + x12_duration),
        ('xminus', physical_qubits[1], x_duration + x12_duration),
    ])

    pms = {}

    pms['layout'] = generate_layout_passmanager(physical_qubits, backend.coupling_map)

    basis_gates = backend.basis_gates + ['rz12', 'x12', 'xplus', 'xminus', 'qutrit_qubit_cx']
    pms['pretranslation'] = generate_translation_passmanager(basis_gates)
    cx_decomp = ReverseCXDecomposition(instruction_durations)
    cx_decomp.calibrations = calibrations
    pms['pretranslation'].append(cx_decomp)
    if do_dd:
        pms['pretranslation'].append(ReverseCXDynamicalDecoupling(backend.target))

    if do_phase_corr:
        pms['phase_corr'] = PassManager([ALAPScheduleAnalysis(instruction_durations)])
        refocusing = QutritToffoliRefocusing()
        refocusing.calibrations = calibrations
        pms['phase_corr'].append(refocusing)
        if do_dd:
            dd = QutritToffoliDynamicalDecoupling(backend.target)
            dd.calibrations = calibrations
            pms['phase_corr'].append(dd)

    # Need to translate + clear the node_start_time dict because transpiler passes try to save the
    # node times to the output circuit at the end of each __call__ and translation of xplus corrupts
    # this dictionary
    basis_gates = backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    if do_phase_corr:
        translation_pass = TranslateAndClearTiming(sel, basis_gates)
    else:
        translation_pass = BasisTranslator(sel, basis_gates)
    pms['translation'] = PassManager([translation_pass])
    pms['layin'] = PassManager([UndoLayout(physical_qubits)])

    return StagedPassManager(list(pms.keys()), **pms)
