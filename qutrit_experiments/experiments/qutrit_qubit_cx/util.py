from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import Calibrations

twopi = 2. * np.pi


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
