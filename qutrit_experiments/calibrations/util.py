"""Common utility functions for calibration definitions."""

from collections.abc import Sequence
from typing import Optional, Union
from qiskit.providers import Backend
from qiskit.pulse import PulseError, Schedule, ScheduleBlock
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.exceptions import CalibrationError
from qiskit.circuit.parameterexpression import ParameterExpression
ParameterValueType = Union[ParameterExpression, float]

def get_operational_qubits(
    backend: Backend,
    qubits: Optional[Sequence[int]] = None
) -> set[int]:
    try:
        faulty_qubits = set(backend.properties().faulty_qubits())
    except AttributeError:
        # Simulator
        faulty_qubits = set()
    if not qubits:
        qubits = range(backend.num_qubits)
    return set(qubits) - faulty_qubits


def get_default_ecr_schedule(backend: Backend, qubits: tuple[int, int]) -> Schedule:
    for gate in ['cx', 'ecr']:
        try:
            return backend.defaults().instruction_schedule_map.get(gate, qubits)
        except PulseError:
            pass
    return None


def get_qutrit_freq_shift(
    qubit: int,
    target: Target,
    calibrations: Calibrations
) -> float:
    f12 = calibrations.get_parameter_value('f12', qubit)
    f01 = target.qubit_properties[qubit].frequency
    return (f12 - f01) * target.dt


def get_qutrit_pulse_gate(
    gate_name: str,
    qubit: int,
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    if not freq_shift:
        freq_shift = get_qutrit_freq_shift(qubit, target, calibrations)
    assign_params_dict = {'freq': freq_shift}
    if assign_params:
        assign_params_dict.update(assign_params)

    return calibrations.get_schedule(gate_name, qubit, assign_params=assign_params_dict,
                                     group=group)


def _collect_qutrit_gate_refs(
    template: ScheduleBlock,
    physical_qubits: tuple[int, ...]
) -> set[tuple[str, int]]:
    ref_keys = set()
    for ref in template.references.unassigned():
        # ref = (gate_name, 'qi', 'qj', ...)
        if (gate := ref[0]) in ['x12', 'sx12']:
            ref_keys.add((gate, physical_qubits[int(ref[1][1:])]))
    return ref_keys


def get_qutrit_qubit_composite_gate(
    gate_name: str,
    physical_qubits: tuple[int, int],
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    freq_shifts = None
    if freq_shift:
        freq_shifts = {physical_qubits[0]: freq_shift}
    return get_qutrit_composite_gate(gate_name, physical_qubits, calibrations,
                                     freq_shifts=freq_shifts, target=target,
                                     assign_params=assign_params, group=group)


def get_qutrit_composite_gate(
    gate_name: str,
    physical_qubits: tuple[int, ...],
    calibrations: Calibrations,
    freq_shifts: Optional[dict[int, float]] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    if not freq_shifts:
        freq_shifts = {}
        for qubit in physical_qubits:
            try:
                freq_shifts[qubit] = get_qutrit_freq_shift(qubit, target, calibrations)
            except CalibrationError:
                pass

    physical_qubits = tuple(physical_qubits)
    template = calibrations.get_template(gate_name, physical_qubits)
    ref_keys = _collect_qutrit_gate_refs(template, physical_qubits)
    assign_params_dict = {('freq', (qubit,), gate): freq_shifts[qubit] for gate, qubit in ref_keys}
    if assign_params:
        assign_params_dict.update(assign_params)

    return calibrations.get_schedule(gate_name, physical_qubits, assign_params=assign_params_dict,
                                     group=group)