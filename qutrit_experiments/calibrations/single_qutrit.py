"""Functions to generate the Calibrations object for qutrit experiments."""

import logging
from typing import Optional
from qiskit import pulse
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from .util import get_operational_qubits
from ..pulse_library import ModulatedDrag

logger = logging.getLogger(__name__)


def make_single_qutrit_gate_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None
) -> Calibrations:
    """Define parameters and schedules for single-qutrit gates."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)

    set_f12_default(backend, calibrations)
    add_x12_sx12(backend, calibrations)
    set_xstark_sxstark_default(backend, calibrations)

    return calibrations


def set_f12_default(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    """Give default values to f12."""
    if 'f12' not in set(p.name for p in calibrations.parameters.keys()):
        calibrations._register_parameter(Parameter('f12'), ())

    operational_qubits = get_operational_qubits(backend)
    for qubit in operational_qubits:
        qubit_props = backend.qubit_properties(qubit)
        freq_12_est = qubit_props.frequency
        try:
            # IBMQubitProperties
            freq_12_est += qubit_props.anharmonicity
        except AttributeError:
            # SingleTransmonQutritBackend
            freq_12_est += backend.anharmonicity

        # Explicitly construct ParameterValue so the time stamp will be 1970-01-01
        calibrations.add_parameter_value(ParameterValue(freq_12_est), 'f12', qubits=[qubit])


def add_x12_sx12(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    r"""X and SX pulses are assumed to effect unitaries of form

    .. math::

        U_{x}(\theta; \delta) = \begin{pmatrix}
                                \cos \frac{\theta}{2}    & -i \sin \frac{\theta} & 0 \\
                                -i \sin \frac{\theta}{2} & \cos \frac{\theta}{2} & 0 \\
                                0                        & 0                     & e^{-i\delta/2}
                                \end{pmatrix},

    and for Ξ and SΞ pulses

    .. math::

        U_{\xi}(\theta; \delta) = \begin{pmatrix}
                                  e^{-i\delta/2} & 0                        & 0 \\
                                  0              & \cos \frac{\theta}{2}    & -i \sin \frac{\theta} \\
                                  0              & -i \sin \frac{\theta}{2} & \cos \frac{\theta}{2}
                                  \end{pmatrix},

    with $\theta=\pi$ or $\pi/2$. The Stark shift-induced phase $\delta$ as well as the geometric
    phases can be corrected with P_0 and P_2 gates. The final
    :math:`X/SX/\Xi/S\Xi` gates are therefore

    .. math::

        X = P2(\delta_X/2 - \pi/2) U_{x}(\pi; \delta_X) \\
        SX = P2(\delta_{SX}/2 - \pi/4) U_{x}(\pi/2; \delta_{SX}) \\
        \Xi = P0(\delta_{\Xi}/2 - \pi/2) U_{\xi}(\pi; \delta_{\Xi}) \\
        S\Xi = P0(\delta_{S\Xi}/2 - \pi/4) U_{\xi}(\pi/2; \delta_{S\Xi}).

    The phase shifts will be taken care of during transpilation.
    """
    drive_channel = pulse.DriveChannel(Parameter('ch0'))

    # Schedules
    for gate_name, pulse_name in [('x12', 'Ξp'), ('sx12', 'Ξ90p')]:
        with pulse.build(name=gate_name) as sched:
            pulse.play(ModulatedDrag(Parameter('duration'), Parameter('amp'),
                                     Parameter('sigma'), Parameter('beta'), Parameter('freq'),
                                     angle=Parameter('angle'), name=pulse_name),
                       drive_channel)
        calibrations.add_schedule(sched, num_qubits=1)

    # Parameter default values
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend)
    for gate_name, qubit_gate_name in [('x12', 'x'), ('sx12', 'sx')]:
        for qubit in operational_qubits:
            qubit_sched = inst_map.get(qubit_gate_name, qubit)
            qubit_pulse = next(inst.pulse for _, inst in qubit_sched.instructions
                               if isinstance(inst, pulse.Play))
            for param_name in ['duration', 'sigma']:
                value = getattr(qubit_pulse, param_name)
                calibrations.add_parameter_value(ParameterValue(value), param_name, qubits=[qubit],
                                                 schedule=gate_name)
            for param_name in ['amp', 'beta', 'freq', 'angle']:
                calibrations.add_parameter_value(ParameterValue(0.), param_name, qubits=[qubit],
                                                 schedule=gate_name)

    # Stark delta parameters
    for name in ['x12stark', 'sx12stark']:
        if name not in set(p.name for p in calibrations.parameters.keys()):
            calibrations._register_parameter(Parameter(name), ())

        for qubit in operational_qubits:
            calibrations.add_parameter_value(ParameterValue(0.), name, qubits=[qubit])


def set_xstark_sxstark_default(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    # Stark delta parameters
    operational_qubits = get_operational_qubits(backend)
    for name in ['xstark', 'sxstark']:
        if name not in set(p.name for p in calibrations.parameters.keys()):
            calibrations._register_parameter(Parameter(name), ())

        for qubit in operational_qubits:
            calibrations.add_parameter_value(ParameterValue(0.), name, qubits=[qubit])


def get_qutrit_pulse_gate(
    gate_name: str,
    qubit: int,
    backend: Backend,
    calibrations: Calibrations,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    qubit_props = backend.qubit_properties(qubit)
    freq = calibrations.get_parameter_value('f12', qubit) - qubit_props.frequency
    assign_params_dict = {'freq': freq}
    if assign_params:
        assign_params_dict.update(assign_params)

    return calibrations.get_schedule(gate_name, qubit, assign_params=assign_params_dict,
                                     group=group)
