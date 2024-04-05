"""Functions to generate the Calibrations object for qutrit experiments."""

from collections.abc import Sequence
import logging
from typing import Optional
import numpy as np
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.circuit import Parameter
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from ..constants import LO_SIGN
from ..gates import ParameterValueType
from ..pulse_library import ModulatedDrag
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_operational_qubits, get_qutrit_freq_shift

logger = logging.getLogger(__name__)


def make_single_qutrit_gate_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None,
    qubits: Optional[Sequence[int]] = None
) -> Calibrations:
    """Define parameters and schedules for single-qutrit gates."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    set_f12_default(backend, calibrations, qubits=qubits)
    add_x12_sx12(backend, calibrations, qubits=qubits)

    return calibrations


def set_f12_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Give default values to f12."""
    if 'f12' not in set(p.name for p in calibrations.parameters.keys()):
        calibrations._register_parameter(Parameter('f12'), ())

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
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
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
    """
    drive_channel = pulse.DriveChannel(Parameter('ch0'))

    # Schedules
    for gate_name, pulse_name in [('x12', 'Ξp'), ('sx12', 'Ξ90p')]:
        with pulse.build(name=gate_name) as sched:
            pulse.play(ModulatedDrag(Parameter('duration'), Parameter('amp'),
                                     Parameter('sigma'), Parameter('beta'),
                                     Parameter('freq') * backend.dt,
                                     angle=Parameter('angle'), name=pulse_name),
                       drive_channel, name=pulse_name)
        calibrations.add_schedule(sched, num_qubits=1)

    # Parameter default values and phase corrections
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)

    for gate_name, pulse_name, qubit_gate_name, geom_phase in [
        ('x12', 'Ξp', 'x', np.pi / 2.),
        ('sx12', 'Ξ90p', 'sx', np.pi / 4.)
    ]:
        for qubit in operational_qubits:
            # Parameter default values
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

            # Phase correction schedules (defined for each qubit because the number of Rz channels is
            # qubit dependent)
            rz_channels = [inst.channel for _, inst in inst_map.get('rz', qubit).instructions]

            delta = Parameter('delta')
            with pulse.build(name=f'{gate_name}_phase_corr') as sched:
                for channel in rz_channels:
                    pulse.shift_phase(LO_SIGN * (geom_phase - delta / 2.), channel)
            calibrations.add_schedule(sched, qubits=[qubit])

            calibrations.add_parameter_value(ParameterValue(0.), 'delta', qubits=[qubit],
                                             schedule=sched.name)

            delta = Parameter('delta')
            with pulse.build(name=f'{qubit_gate_name}_phase_corr') as sched:
                for channel in rz_channels:
                    pulse.shift_phase(LO_SIGN * (delta / 2. - geom_phase), channel)
            calibrations.add_schedule(sched, qubits=[qubit])

            calibrations.add_parameter_value(ParameterValue(0.), 'delta', qubits=[qubit],
                                             schedule=sched.name)

