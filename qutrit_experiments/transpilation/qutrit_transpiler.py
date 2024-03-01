"""Transpilation functions."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations

from ..gates import QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES
from .custom_pulses import ConvertCustomPulses
from .qutrit_circuits import ContainsQutritInstruction, AddQutritCalibrations
from .rz import ConsolidateRZAngle, CorrectRZSign


@dataclass
class QutritTranspileOptions:
    """Options for qutrit transpilation."""
    use_waveform: bool = False
    remove_custom_pulses: bool = True
    resolve_rz: Optional[list[str]] = None
    consolidate_rz: bool = True


def make_instruction_durations(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> InstructionDurations:
    """Construct an InstructionDurations object including qutrit gate durations."""
    if qubits is None:
        qubits = set(range(backend.num_qubits)) - set(backend.properties().faulty_qubits())

    instruction_durations = InstructionDurations(backend.instruction_durations, dt=backend.dt)
    for inst in QUTRIT_PULSE_GATES:
        durations = [(inst.gate_name, qubit,
                      calibrations.get_schedule(inst.gate_name, qubit).duration)
                     for qubit in qubits]
        instruction_durations.update(durations)
    for inst in QUTRIT_VIRTUAL_GATES:
        instruction_durations.update([(inst.gate_name, qubit, 0) for qubit in qubits])
    return instruction_durations


def transpile_qutrit_circuits(
    circuits: Union[QuantumCircuit, list[QuantumCircuit]],
    backend: Backend,
    calibrations: Calibrations,
    instruction_durations: Optional[InstructionDurations] = None,
    options: Optional[QutritTranspileOptions] = None
) -> list[QuantumCircuit]:
    """Recompute the gate durations, calculate the phase shifts for all qutrit gates, and insert
    AC Stark shift corrections to qubit gates"""
    if instruction_durations is None:
        instruction_durations = make_instruction_durations(backend, calibrations)
    if options is None:
        options = QutritTranspileOptions()

    def contains_qutrit_gate(property_set):
        return property_set['contains_qutrit_gate']

    pm = PassManager()
    pm.append(CorrectRZSign())
    pm.append(ContainsQutritInstruction())
    scheduling = ALAPScheduleAnalysis(instruction_durations)
    add_cal = AddQutritCalibrations(backend.target, backend.configuration().channels,
                                    resolve_rz=options.resolve_rz)
    add_cal.calibrations = calibrations # See the comment in the class for why we do this
    pm.append([scheduling, add_cal], condition=contains_qutrit_gate)
    if options.use_waveform:
        pm.append(ConvertCustomPulses(options.remove_custom_pulses))
    if options.consolidate_rz:
        pm.append(ConsolidateRZAngle())
    return pm.run(circuits)
