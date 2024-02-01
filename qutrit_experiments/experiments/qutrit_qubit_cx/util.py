from collections.abc import Sequence
from enum import IntEnum
from typing import Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.framework import BackendData, BackendTiming

from ...gates import X12Gate

twopi = 2. * np.pi


class RCRType(IntEnum):
    X = 2 # CRCR angle = 2 * (θ_0 + θ_1 - 2*θ_2)
    X12 = 0 # CRCR angle = 2 * (θ_1 + θ_2 - 2*θ_0)


def make_rcr_circuit(
    physical_qubits: Sequence[int],
    cr_schedule: ScheduleBlock,
    rcr_type: RCRType
) -> QuantumCircuit:
    params = cr_schedule.parameters

    rcr_circuit = QuantumCircuit(2)
    if rcr_type == RCRType.X:
        rcr_circuit.x(0)
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.x(0)
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
    else:
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])

    rcr_circuit.add_calibration('cr', physical_qubits, cr_schedule, params)

    return rcr_circuit


def make_crcr_circuit(
    physical_qubits: Sequence[int],
    cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
    rx_schedule: Union[ScheduleBlock, None],
    rcr_type: RCRType
) -> QuantumCircuit:
    crp_params = cr_schedules[0].parameters
    crm_params = cr_schedules[1].parameters
    if rx_schedule is not None:
        rx_params = rx_schedule.parameters

    crcr_circuit = QuantumCircuit(2)
    if rcr_type == RCRType.X:
        # [X+]-[RCR-]
        if rx_schedule is not None:
            crcr_circuit.append(Gate('offset_rx', 1, rx_params), [1])
        crcr_circuit.append(X12Gate())
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.x(0)
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        # [X+]-[RCR+] x 2
        for _ in range(2):
            crcr_circuit.append(X12Gate())
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.x(0)
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
    else:
        # [RCR+]-[X+] x 2
        for _ in range(2):
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.append(X12Gate(), [0])
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.x(0)
        # [RCR-]-[X+]
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.append(X12Gate(), [0])
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.x(0)
        if rx_schedule is not None:
            crcr_circuit.append(Gate('offset_rx', 1, rx_params), [1])

    crcr_circuit.add_calibration('crp', physical_qubits, cr_schedules[0], crp_params)
    crcr_circuit.add_calibration('crm', physical_qubits, cr_schedules[1], crm_params)
    if rx_schedule is not None:
        crcr_circuit.add_calibration('offset_rx', (physical_qubits[1],), rx_schedule, rx_params)

    return crcr_circuit


def get_margin(
    cr_schedule: ScheduleBlock,
    width_param_name: str,
    margin_param_name: str,
    widths: Union[float, Sequence[float]],
    backend: Backend
):
    width_param = cr_schedule.get_parameters(width_param_name)[0]
    margin_param = cr_schedule.get_parameters(margin_param_name)[0]
    test_assign = {width_param: 0., margin_param: 0.}
    risefall_duration = cr_schedule.assign_parameters(test_assign, inplace=False).duration
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
