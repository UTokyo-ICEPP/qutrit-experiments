from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BackendData, BackendTiming

from ...gates import CrossResonanceGate, CrossResonanceMinusGate, CrossResonancePlusGate, X12Gate

twopi = 2. * np.pi


class RCRType(IntEnum):
    X = 2 # CRCR angle = 2 * (θ_0 + θ_1 - 2*θ_2)
    X12 = 0 # CRCR angle = 2 * (θ_1 + θ_2 - 2*θ_0)


def make_cr_circuit(
    physical_qubits: Sequence[int],
    arg: Union[ScheduleBlock, Calibrations],
    param_cal_groups: Optional[dict[str, str]] = None
) -> QuantumCircuit:
    if isinstance(arg, Calibrations):
        assign_params = None
        if param_cal_groups:
            assign_params = {pname: arg.get_parameter_value(pname, physical_qubits, 'cr',
                                                            group=group)
                             for pname, group in param_cal_groups.items()}
        cr_schedule = arg.get_schedule('cr', physical_qubits, assign_params=assign_params)
    else:
        cr_schedule = arg

    params = cr_schedule.parameters
    circuit = QuantumCircuit(2)
    circuit.append(CrossResonanceGate(params), [0, 1])
    circuit.add_calibration(CrossResonanceGate.gate_name, physical_qubits, cr_schedule, params)

    return circuit


def make_rcr_circuit(
    physical_qubits: Sequence[int],
    arg: Union[ScheduleBlock, Calibrations],
    rcr_type: Optional[RCRType] = None,
    param_cal_groups: Optional[dict[str, str]] = None
) -> QuantumCircuit:
    if isinstance(arg, Calibrations):
        assign_params = None
        if param_cal_groups:
            assign_params = {pname: arg.get_parameter_value(pname, physical_qubits, 'cr',
                                                            group=group)
                             for pname, group in param_cal_groups.items()}
        cr_schedule = arg.get_schedule('cr', physical_qubits, assign_params=assign_params)
        rcr_type = RCRType(arg.get_parameter_value('rcr_type', physical_qubits))
    else:
        cr_schedule = arg

    params = cr_schedule.parameters
    rcr_circuit = QuantumCircuit(2)
    if rcr_type == RCRType.X:
        rcr_circuit.x(0)
        rcr_circuit.append(CrossResonanceGate(params), [0, 1])
        rcr_circuit.x(0)
        rcr_circuit.append(CrossResonanceGate(params), [0, 1])
    else:
        rcr_circuit.append(CrossResonanceGate(params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])
        rcr_circuit.append(CrossResonanceGate(params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])

    rcr_circuit.add_calibration(CrossResonanceGate.gate_name, physical_qubits, cr_schedule, params)

    return rcr_circuit


def make_crcr_circuit(
    physical_qubits: Sequence[int],
    arg: Union[tuple[ScheduleBlock, ScheduleBlock], Calibrations],
    rx_angle: Optional[float] = None,
    rcr_type: Optional[RCRType] = None,
    param_cal_groups: Optional[dict[str, str]] = None
) -> QuantumCircuit:
    if isinstance(arg, Calibrations):
        rx_group = 'default'
        if param_cal_groups:
            param_cal_groups = dict(param_cal_groups)
            try:
                rx_group = param_cal_groups.pop('qutrit_qubit_cx_offsetrx')
            except KeyError:
                pass
        cr_schedules = get_cr_schedules(arg, physical_qubits, param_cal_groups=param_cal_groups)
        rx_angle = arg.get_parameter_value('qutrit_qubit_cx_offsetrx', physical_qubits[1],
                                           group=rx_group)
        rcr_type = RCRType(arg.get_parameter_value('rcr_type', physical_qubits))
    else:
        cr_schedules = arg

    crp_params = cr_schedules[0].parameters
    crm_params = cr_schedules[1].parameters

    crcr_circuit = QuantumCircuit(2)
    if rcr_type == RCRType.X:
        # [X+]-[RCR-]
        if rx_angle:
            crcr_circuit.rz(np.pi / 2., 1)
            crcr_circuit.sx(1)
            crcr_circuit.rz(rx_angle + np.pi, 1)
            crcr_circuit.sx(1)
            crcr_circuit.rz(np.pi / 2., 1)
        crcr_circuit.append(X12Gate(), [0])
        crcr_circuit.append(CrossResonanceMinusGate(crm_params), [0, 1])
        crcr_circuit.x(0)
        crcr_circuit.append(CrossResonanceMinusGate(crm_params), [0, 1])
        # [X+]-[RCR+] x 2
        for _ in range(2):
            crcr_circuit.append(X12Gate(), [0])
            crcr_circuit.append(CrossResonancePlusGate(crp_params), [0, 1])
            crcr_circuit.x(0)
            crcr_circuit.append(CrossResonancePlusGate(crp_params), [0, 1])
    else:
        # [RCR+]-[X+] x 2
        for _ in range(2):
            crcr_circuit.append(CrossResonancePlusGate(crp_params), [0, 1])
            crcr_circuit.append(X12Gate(), [0])
            crcr_circuit.append(CrossResonancePlusGate(crp_params), [0, 1])
            crcr_circuit.x(0)
        # [RCR-]-[X+]
        crcr_circuit.append(CrossResonanceMinusGate(crm_params), [0, 1])
        crcr_circuit.append(X12Gate(), [0])
        crcr_circuit.append(CrossResonanceMinusGate(crm_params), [0, 1])
        crcr_circuit.x(0)
        if rx_angle:
            crcr_circuit.rz(np.pi / 2., 1)
            crcr_circuit.sx(1)
            crcr_circuit.rz(rx_angle + np.pi, 1)
            crcr_circuit.sx(1)
            crcr_circuit.rz(np.pi / 2., 1)

    crcr_circuit.add_calibration(CrossResonancePlusGate.gate_name, physical_qubits,
                                 cr_schedules[0], crp_params)
    crcr_circuit.add_calibration(CrossResonanceMinusGate.gate_name, physical_qubits,
                                 cr_schedules[1], crm_params)

    return crcr_circuit


def get_margin(
    risefall_duration: float,
    widths: Union[float, Sequence[float]],
    backend: Backend
):
    backend_timing = BackendTiming(backend)
    granularity = BackendData(backend).granularity
    supports = np.asarray(widths) + risefall_duration
    if (is_scalar := (supports.ndim == 0)):
        supports = [supports]

    margins = np.array([backend_timing.round_pulse(samples=support) - support
                        for support in supports])
    margins = np.where(margins < 0., margins + granularity, margins)
    if is_scalar:
        return margins[0]
    else:
        return margins


def get_cr_schedules(
    calibrations: Calibrations,
    qubits: Sequence[int],
    free_parameters: Optional[list[str]] = None,
    assign_params: Optional[dict[str, Any]] = None,
    param_cal_groups: Optional[dict[str, str]] = None
) -> tuple[ScheduleBlock, ScheduleBlock]:
    """Return a tuple of crp and crm schedules with optional free parameters"""
    if assign_params:
        assign_params = dict(assign_params)
    else:
        assign_params = {}
    if free_parameters:
        assign_params.update({pname: Parameter(pname) for pname in free_parameters})
    if param_cal_groups:
        assign_params.update({pname: calibrations.get_parameter_value(pname, qubits, 'cr',
                                                                      group=group)
                              for pname, group in param_cal_groups.items()})

    cr_schedules = [calibrations.get_schedule('cr', qubits, assign_params=assign_params)]

    for pname in ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']:
        # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
        assign_params.setdefault(pname, 0.)
        assign_params[pname] += np.pi
    cr_schedules.append(calibrations.get_schedule('cr', qubits, assign_params=assign_params))

    return tuple(cr_schedules)
