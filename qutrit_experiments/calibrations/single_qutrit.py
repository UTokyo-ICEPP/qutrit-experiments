"""Functions to generate the Calibrations object for qutrit experiments."""

from collections.abc import Sequence
import logging
from typing import Optional
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScalableSymbolicPulse
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from ..pulse_library import ModulatedDrag
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_operational_qubits

logger = logging.getLogger(__name__)


def make_single_qutrit_gate_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None,
    set_defaults: bool = True,
    qubits: Optional[Sequence[int]] = None,
) -> Calibrations:
    """Define parameters and schedules for single-qutrit gates.

    Args:
        backend: Backend to use.
        calibrations: Calibrations object. If not given, a new instance is created from the backend.
        set_defaults: If True (default), sets the default values of the parameters.
        qubits: Qubits to set the default parameter values on. If not given, all qubits in the
            backend will be used.

    Returns:
        The passed or newly created Calibrations instance.
    """
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    if 'f12' not in set(p.name for p in calibrations.parameters.keys()):
        calibrations._register_parameter(Parameter('f12'), ())
    add_x12_sx12(calibrations)

    if set_defaults:
        set_f12_default(backend, calibrations, qubits=qubits)
        set_x12_sx12_default(backend, calibrations, qubits=qubits)

    return calibrations


def set_f12_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Give default values to f12.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
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

        U_{\xi}(\theta; \delta) =
        \begin{pmatrix}
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

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    drive_channel = pulse.DriveChannel(Parameter('ch0'))

    # Schedules
    for gate_name, pulse_name in [('x12', 'Ξp'), ('sx12', 'Ξ90p')]:
        with pulse.build(name=gate_name) as sched:
            pulse.play(ModulatedDrag(Parameter('duration'), Parameter('amp'), Parameter('sigma'),
                                     Parameter('beta'), Parameter('freq'),
                                     angle=Parameter('angle'), name=pulse_name),
                       drive_channel, name=pulse_name)
        calibrations.add_schedule(sched, num_qubits=1)
        # Define the Stark phase shift without a containing schedule
        calibrations._register_parameter(Parameter(f'delta_{gate_name}'), ())

    for gate_name in ['x', 'sx']:
        calibrations._register_parameter(Parameter(f'delta_{gate_name}'), ())


def set_x12_sx12_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Set the default values for x12 and sx12 calibrations.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    # Parameter default values and phase corrections
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)

    for gate_name, qubit_gate_name in [('x12', 'x'), ('sx12', 'sx')]:
        for qubit in operational_qubits:
            # Parameter default values
            qubit_sched = inst_map.get(qubit_gate_name, qubit)
            qubit_pulse = next(inst.pulse for _, inst in qubit_sched.instructions
                               if isinstance(inst, pulse.Play))
            if not (isinstance(qubit_pulse, ScalableSymbolicPulse)
                    and qubit_pulse.pulse_type == 'Drag'):
                raise RuntimeError(f'Pulse of q{qubit} {qubit_gate_name} is not Drag')
            for param_name in ['duration', 'sigma']:
                value = getattr(qubit_pulse, param_name)
                calibrations.add_parameter_value(ParameterValue(value), param_name, qubits=[qubit],
                                                 schedule=gate_name)
            # Give a default value of A01 / sqrt(2) for amplitude (rough amp cal requires a nonzero
            # initial value for the amplitude)
            calibrations.add_parameter_value(ParameterValue(qubit_pulse.amp / np.sqrt(2.)), 'amp',
                                             qubits=[qubit], schedule=gate_name)
            for param_name in ['beta', 'freq', 'angle']:
                calibrations.add_parameter_value(ParameterValue(0.), param_name, qubits=[qubit],
                                                 schedule=gate_name)

            calibrations.add_parameter_value(ParameterValue(0.), f'delta_{gate_name}',
                                             qubits=[qubit])
            calibrations.add_parameter_value(ParameterValue(0.), f'delta_{qubit_gate_name}',
                                             qubits=[qubit])
