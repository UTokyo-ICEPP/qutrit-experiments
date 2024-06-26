"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, BasisTranslator, TimeUnitConversion
from qiskit_experiments.calibration_management import Calibrations

from ..transpilation.custom_pulses import AttachCalibration
from ..transpilation.qutrit_qubit_cx import ReverseCXDecomposition
from ..transpilation.qutrit_toffoli import QutritMCGateDecomposition, QutritToffoliRefocusing
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.layout_and_translation import (UndoLayout, TranslateAndClearTiming,
                                                    generate_layout_passmanager)
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
    """Return a pass manager to translate a qubit-qutrit-qubit circuit containing QutritCCXGate or
    something similar."""
    physical_qubits = tuple(physical_qubits)

    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits[-2:])

    ecr_based = 'ecr' in backend.basis_gates

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
    if not ecr_based:
        ecrs = []
        for name, qubits in instruction_durations.duration_by_name_qubits:
            if name == 'cx' and set(qubits) <= set(physical_qubits):
                ecr_duration = calibrations.get_schedule('ecr', qubits).duration
                ecrs.append(('ecr', qubits, ecr_duration))
        instruction_durations.update(ecrs)

    pms = {
        'layout': generate_layout_passmanager(physical_qubits, backend.coupling_map)
    }

    decomp_pass = QutritMCGateDecomposition(instruction_durations, do_dd,
                                            backend.target.pulse_alignment)
    decomp_pass.calibrations = calibrations
    pms['pretranslation'] = PassManager([decomp_pass])
    if do_phase_corr:
        decomp_pass = ReverseCXDecomposition(instruction_durations, do_dd,
                                             backend.target.pulse_alignment)
        decomp_pass.calibrations = calibrations
        pms['pretranslation'].append(decomp_pass)
    basis_gates = backend.basis_gates + ['rz12', 'x12', 'xplus', 'xminus']
    if rcr_type != QutritQubitCXType.REVERSE:
        basis_gates.append('qutrit_qubit_cx')
    if not ecr_based:
        basis_gates.remove('cx')
        basis_gates.append('ecr')
        attach_pass = AttachCalibration(['ecr'])
        attach_pass.calibrations = calibrations
        pms['pretranslation'].append(attach_pass)
    pms['pretranslation'].append(BasisTranslator(sel, basis_gates))

    if do_phase_corr:
        refocusing_pass = QutritToffoliRefocusing(instruction_durations, do_dd,
                                                  backend.target.pulse_alignment)
        refocusing_pass.calibrations = calibrations
        pms['pretranslation'].append([
            TimeUnitConversion(inst_durations=instruction_durations),
            ALAPScheduleAnalysis(instruction_durations),
            refocusing_pass
        ])
    translation_passes = []
    # Need to translate + clear the node_start_time dict because transpiler passes try to save the
    # node times to the output circuit at the end of each __call__ and translation of xplus corrupts
    # this dictionary
    basis_gates = backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    if not ecr_based:
        basis_gates.remove('cx')
        basis_gates.append('ecr')
    translation_passes.append(TranslateAndClearTiming(sel, basis_gates))
    pms['translation'] = PassManager(translation_passes)
    pms['layin'] = PassManager([UndoLayout(physical_qubits)])

    return StagedPassManager(list(pms.keys()), **pms)
