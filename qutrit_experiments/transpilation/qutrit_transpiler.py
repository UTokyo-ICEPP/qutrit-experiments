"""Transpilation functions."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.exceptions import CalibrationError

from ..constants import LO_SIGN
from ..gates import QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES
from .custom_pulses import ConvertCustomPulses
from .qutrit_circuits import ContainsQutritInstruction, AddQutritCalibrations
from .rz import CastRZToAngle, ConsolidateRZAngle, InvertRZSign


@dataclass
class QutritTranspileOptions:
    """Options for qutrit transpilation."""
    use_waveform: bool = False
    remove_custom_pulses: bool = True
    rz_casted_gates: list[str] = None
    consolidate_rz: bool = True

    def __post_init__(self):
        if self.rz_casted_gates is None:
            self.rz_casted_gates = []


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
        durations = []
        match inst.num_qubits:
            case 1:
                for qubit in qubits:
                    try:
                        duration = calibrations.get_schedule(inst.gate_name, qubit).duration
                    except CalibrationError:
                        continue
                    durations.append((inst.gate_name, qubit, duration))
            case 2:
                for edge in backend.coupling_map.get_edges():
                    if edge[0] in qubits and edge[1] in qubits:
                        try:
                            duration = calibrations.get_schedule(inst.gate_name, edge).duration
                        except CalibrationError:
                            continue
                        durations.append((inst.gate_name, edge, duration))

        instruction_durations.update(durations)
    for inst in QUTRIT_VIRTUAL_GATES:
        match inst.num_qubits:
            case 1:
                durations = [(inst.gate_name, qubit, 0) for qubit in qubits]
            case 2:
                durations = []
                for edge in backend.coupling_map.get_edges():
                    if edge[0] in qubits and edge[1] in qubits:
                        durations.append((inst.gate_name, edge, 0))
        instruction_durations.update(durations)

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
    pm.append(ContainsQutritInstruction())
    scheduling = ALAPScheduleAnalysis(instruction_durations)
    add_cal = AddQutritCalibrations(backend.target)
    add_cal.calibrations = calibrations # See the comment in the class for why we do this
    pm.append([scheduling, add_cal], condition=contains_qutrit_gate)
    if LO_SIGN > 0.:
        pm.append(InvertRZSign())
    if options.rz_casted_gates:
        pm.append(CastRZToAngle(backend.target.instruction_schedule_map(),
                                backend.configuration().channels, options.rz_casted_gates))
    if options.consolidate_rz:
        pm.append(ConsolidateRZAngle())
    if options.use_waveform:
        pm.append(ConvertCustomPulses(options.remove_custom_pulses))

    return pm.run(circuits)
