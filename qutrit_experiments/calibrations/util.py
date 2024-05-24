"""Common utility functions for calibration definitions."""

from collections.abc import Sequence
from typing import Optional, Union
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.exceptions import CalibrationError
ParameterValueType = Union[ParameterExpression, float]


def get_operational_qubits(
    backend: Backend,
    qubits: Optional[Sequence[int]] = None
) -> set[int]:
    """Return a set of ids of operational qubits on the backend.

    Args:
        backend: Backend.
        qubits: If given, return only the subset of this list.
    """
    try:
        faulty_qubits = set(backend.properties().faulty_qubits())
    except AttributeError:
        # Simulator
        faulty_qubits = set()
    if not qubits:
        qubits = range(backend.num_qubits)
    return set(qubits) - faulty_qubits


def get_qutrit_freq_shift(
    qubit: int,
    target: Target,
    calibrations: Calibrations
) -> float:
    """Return the frequency difference between EF and GE resonances in 1/dt.

    Args:
        qubit: The qubit id.
        target: Backend transpilation target.
        calibrations: Calibrations object where f12 of the qubit is defined.
    """
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
    """Return the ScheduleBlock for the single-qutrit gate of the given name.

    The returned schedule will have the ``freq`` parameter set to the ``freq_shift`` value.

    Args:
        gate_name: Name of the single-qutrit gate.
        qubit: Qubit id of the qutrit.
        calibrations: Calibrations object from which to obtain the schedule for the gate (and the
            f12 value, if ``freq_shift`` is not given).
        freq_shift: EF-GE frequency difference in 1/dt. If not given, ``get_qutrit_freq_shift`` is
            used to obtain the value from ``calibrations``.
        target: Backend transpilation target.
        assign_params: Additional parameter assignments.
        group: Calbiration group name.
    """
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
    """Return the ScheduleBlock for the qutrit-qubit composite gate of the given name.

    The returned schedule will have the ``freq`` parameter set to the ``freq_shift`` value.

    Args:
        gate_name: Name of the single-qutrit gate.
        physical_qubits: Qubit ids of the qutrit and qubit.
        calibrations: Calibrations object from which to obtain the schedule for the gate (and the
            f12 value, if ``freq_shift`` is not given).
        freq_shift: EF-GE frequency difference in 1/dt. If not given, ``get_qutrit_freq_shift`` is
            used to obtain the value from ``calibrations``.
        target: Backend transpilation target.
        assign_params: Additional parameter assignments.
        group: Calbiration group name.
    """
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
    """Return the ScheduleBlock for the qutrit composite gate of the given name.

    The returned schedule will have the ``freq`` parameter set to the ``freq_shift`` value.

    Args:
        gate_name: Name of the single-qutrit gate.
        physical_qubits: Qubit ids on which the gate acts.
        calibrations: Calibrations object from which to obtain the schedule for the gate (and the
            f12 value, if ``freq_shift`` is not given).
        freq_shift: EF-GE frequency difference in 1/dt. If not given, ``get_qutrit_freq_shift`` is
            used to obtain the value from ``calibrations``.
        target: Backend transpilation target.
        assign_params: Additional parameter assignments.
        group: Calbiration group name.
    """
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
