"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import (ALAPScheduleAnalysis, BasisTranslator, PadDynamicalDecoupling,
                                      TimeUnitConversion)
from qiskit_experiments.calibration_management import Calibrations

from ..transpilation.dynamical_decoupling import AddDDCalibration
from ..transpilation.qutrit_qubit_cx import ReverseCXDecomposition, ReverseCXDynamicalDecoupling
from ..transpilation.qutrit_toffoli import (ContainsQutritMCGate, QutritToffoliRefocusing,
                                            QutritToffoliDynamicalDecoupling)
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.layout_and_translation import (UndoLayout, TranslateAndClearTiming,
                                                    generate_layout_passmanager,
                                                    generate_translation_passmanager)
from ..gates import QutritQubitCXType, QutritCCXGate, QutritMCGate


def qutrit_toffoli_circuit(
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int],
    gate: Optional[QutritMCGate] = None,
    do_phase_corr: bool = True,
    do_dd: bool = True
) -> QuantumCircuit:
    pm = qutrit_toffoli_translator(backend, calibrations, physical_qubits,
                                   do_phase_corr=do_phase_corr, do_dd=do_dd)
    if gate is None:
        gate = QutritCCXGate()
    circuit = QuantumCircuit(3)
    circuit.append(gate, [0, 1, 2])
    return pm.run(circuit)


def qutrit_toffoli_translator(
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int],
    do_phase_corr: bool = True,
    do_dd: bool = True
) -> StagedPassManager:
    """Return a pass manager to translate a qubit-qutrit-qubit circuit containing QutritCCXGate or something similar."""
    physical_qubits = tuple(physical_qubits)

    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits[-2:])
    rcr_types = {physical_qubits[-2:]: rcr_type}

    x_duration = calibrations.get_schedule('x', physical_qubits[-2]).duration
    x12_duration = calibrations.get_schedule('x12', physical_qubits[-2]).duration
    instruction_durations = make_instruction_durations(backend, calibrations, physical_qubits)
    instruction_durations.update([
        ('xplus', physical_qubits[-2], x_duration + x12_duration),
        ('xminus', physical_qubits[-2], x_duration + x12_duration),
    ])
    if rcr_type != QutritQubitCXType.REVERSE:
        cr_duration = calibrations.get_schedule('cr', physical_qubits[-2:]).duration
        instruction_durations.update([('cr', physical_qubits[-2:], cr_duration)])

    pms = {
        'layout': generate_layout_passmanager(physical_qubits, backend.coupling_map)
    }

    def contains_qutrit_toffoli(property_set):
        return property_set.get('has_qutrit_mcgate', False)

    basis_gates = backend.basis_gates + ['rz12', 'x12', 'xplus', 'xminus']
    if rcr_type == QutritQubitCXType.REVERSE:
        basis_gates.append('qutrit_qubit_cz')
    else:
        basis_gates.append('qutrit_qubit_cx')

    pms['pretranslation'] = PassManager(
        [
            ContainsQutritMCGate(),
            BasisTranslator(sel, basis_gates),
            ReverseCXDecomposition(instruction_durations, rcr_types)
        ]
    )
    if do_phase_corr or do_dd:
        pms['pretranslation'].append([
            TimeUnitConversion(inst_durations=instruction_durations),
            ALAPScheduleAnalysis(instruction_durations)
        ])
    if do_phase_corr:
        pms['pretranslation'].append(
            QutritToffoliRefocusing(instruction_durations),
            condition=contains_qutrit_toffoli
        )
    if do_dd:
        qubits = list(physical_qubits)
        qubits.pop(-2)
        pms['pretranslation'].append(
            PadDynamicalDecoupling(durations=instruction_durations, dd_sequence=[XGate(), XGate()],
                                   qubits=qubits, spacing=[0.25, 0.5, 0.25],
                                   pulse_alignment=backend.target.pulse_alignment)
        )

    translation_passes = []
    # if do_dd:
    #     add_dd = AddDDCalibration()
    #     add_dd.calibrations = calibrations
    #     translation_passes.append(add_dd)
    # Need to translate + clear the node_start_time dict because transpiler passes try to save the
    # node times to the output circuit at the end of each __call__ and translation of xplus corrupts
    # this dictionary
    basis_gates = backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    translation_passes.append(TranslateAndClearTiming(sel, basis_gates))
    pms['translation'] = PassManager(translation_passes)
    pms['layin'] = PassManager([UndoLayout(physical_qubits)])

    return StagedPassManager(list(pms.keys()), **pms)
