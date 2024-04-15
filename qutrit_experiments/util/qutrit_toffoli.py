"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations

from ..transpilation.qutrit_toffoli import (QutritToffoliDecomposition, QutritToffoliRefocusing,
                                            QutritToffoliDynamicalDecoupling)
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.layout_and_translation import (generate_layout_passmanager,
                                                    generate_translation_passmanager)
from ..gates import QutritQubitCXGate, QutritToffoliGate


def qutrit_toffoli_circuit(
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int]
) -> QuantumCircuit:
    physical_qubits = tuple(physical_qubits)
    x_duration = calibrations.get_schedule('x', physical_qubits[1]).duration
    x12_duration = calibrations.get_schedule('x12', physical_qubits[1]).duration
    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits[1:])

    basis_gates = backend.basis_gates + ['rz12', 'xplus', 'xminus',
                                         QutritQubitCXGate.of_type(rcr_type).gate_name]
    # Apply refocusing and DD
    instruction_durations = make_instruction_durations(backend, calibrations, physical_qubits)
    instruction_durations.update([
        ('xplus', physical_qubits[1], x_duration + x12_duration),
        ('xminus', physical_qubits[1], x_duration + x12_duration),
    ])

    # Decompose the Toffoli gate into high-level gates. Use the BasisTranslator to break down the
    # standard CX
    # If CX type is reverse, decompose into the gate sequence of refocusing (cycled) version
    pretranslation_pm = PassManager()
    decomp = QutritToffoliDecomposition(backend.target, instruction_durations)
    decomp.calibrations = calibrations
    pretranslation_pm.append(decomp)
    pretranslation_pm.append(BasisTranslator(sel, basis_gates))

    phase_corr_pm = PassManager()
    phase_corr_pm.append(ALAPScheduleAnalysis(instruction_durations))
    refocusing = QutritToffoliRefocusing()
    refocusing.calibrations = calibrations
    phase_corr_pm.append(refocusing)
    dd = QutritToffoliDynamicalDecoupling(backend.target)
    dd.calibrations = calibrations
    phase_corr_pm.append(dd)

    pass_manager = StagedPassManager(
        ['layout', 'pretranslation', 'phase_corr'],
        layout=generate_layout_passmanager(physical_qubits, backend.coupling_map),
        pretranslation=pretranslation_pm,
        phase_corr=phase_corr_pm
    )

    # Run the PM
    circuit = QuantumCircuit(3)
    circuit.append(QutritToffoliGate(), [0, 1, 2])
    circuit = pass_manager.run(circuit)

    # Decompose xplus and xminus (need to run separately because node_start_time gets corrupt)
    pm = generate_translation_passmanager(backend.basis_gates
                                          + [g.gate_name for g in BASIS_GATES])
    circuit = pm.run(circuit)

    # Undo layout to backend qubits
    dag = circuit_to_dag(circuit)
    subdag = next(d for d in dag.separable_circuits(remove_idle_qubits=True) if d.size() != 0)
    subdag.calibrations = dag.calibrations
    circuit = dag_to_circuit(subdag)

    # Reorder the qubits if necessary (separable_circuits sorts the qreg with qubit index)
    if (squbits := tuple(sorted(physical_qubits))) != physical_qubits:
        perm = [physical_qubits.index(q) for q in squbits]
        circuit = QuantumCircuit(3).compose(circuit, perm)

    return circuit
