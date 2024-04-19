"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations

from ..transpilation.qutrit_qubit_cx import ReverseCXDecomposition
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
    physical_qubits = tuple(physical_qubits)
    x_duration = calibrations.get_schedule('x', physical_qubits[1]).duration
    x12_duration = calibrations.get_schedule('x12', physical_qubits[1]).duration

    # Apply refocusing and DD
    instruction_durations = make_instruction_durations(backend, calibrations, physical_qubits)
    instruction_durations.update([
        ('x12', physical_qubits[1], x12_duration),
        ('xplus', physical_qubits[1], x_duration + x12_duration),
        ('xminus', physical_qubits[1], x_duration + x12_duration),
    ])

    basis_gates = backend.basis_gates + ['rz12', 'x12', 'xplus', 'xminus', 'qutrit_qubit_cx']
    pretranslation_pm = generate_translation_passmanager(basis_gates)
    cx_decomp = ReverseCXDecomposition(instruction_durations)
    cx_decomp.calibrations = calibrations
    pretranslation_pm.append(cx_decomp)

    phase_corr_pm = PassManager()
    phase_corr_pm.append(ALAPScheduleAnalysis(instruction_durations))
    refocusing = QutritToffoliRefocusing()
    refocusing.calibrations = calibrations
    phase_corr_pm.append(refocusing)
    dd = QutritToffoliDynamicalDecoupling(backend.target)
    dd.calibrations = calibrations
    phase_corr_pm.append(dd)

    # Need to translate + clear the node_start_time dict because transpiler passes try to save the
    # node times to the output circuit at the end of each __call__ and translation of xplus corrupts
    # this dictionary
    basis_gates = backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    translation_pm = PassManager([TranslateAndClearTiming(sel, basis_gates)])

    layin_pm = PassManager([UndoLayout(physical_qubits)])
    
    pass_manager = StagedPassManager(
        ['layout', 'pretranslation', 'phase_corr', 'translation', 'layin'],
        layout=generate_layout_passmanager(physical_qubits, backend.coupling_map),
        pretranslation=pretranslation_pm,
        phase_corr=phase_corr_pm,
        translation=translation_pm,
        layin=layin_pm
    )

    # Run the PM
    circuit = QuantumCircuit(3)
    circuit.append(QutritToffoliGate(), [0, 1, 2])
    circuit = pass_manager.run(circuit)

    return circuit
