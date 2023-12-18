import logging
from typing import Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate, SXGate
from qiskit.pulse.calibration_entries import ScheduleDef
from qiskit.transpiler import Target, InstructionProperties
from qiskit_experiments.calibration_management import Calibrations, ParameterValue
from qiskit_experiments.exceptions import CalibrationError

from .pulse_library import ModulatedDrag

logger = logging.getLogger(__name__)


def make_single_qutrit_gate_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None
) -> Calibrations:
    """"""
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

    for qubit in range(backend.num_qubits):
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

    with $\theta=\pi$ or $\pi/2$. The Stark shift-induced phase $\delta$ can be corrected with
    P_0 and P_2 gates (phase shifts on the drive and qutrit channels).
    """
    drive_channel = pulse.DriveChannel(Parameter('ch0'))

    # Schedules
    for gate_name, pulse_name in [('x12', 'Ξp'), ('sx12', 'Ξ90p')]:
        with pulse.build(name=gate_name) as sched:
            with pulse.phase_offset(Parameter('phase_offset'), drive_channel):
                pulse.play(ModulatedDrag(Parameter('duration'), Parameter('amp'),
                                         Parameter('sigma'), Parameter('beta'), Parameter('freq'),
                                         name=pulse_name),
                           drive_channel)
        calibrations.add_schedule(sched, num_qubits=1)

    # Parameter default values
    inst_map = backend.instruction_schedule_map
    for qubit in range(backend.num_qubits):
        for gate_name, qubit_gate_name in [('x12', 'x'), ('sx12', 'sx')]:
            qubit_sched = inst_map.get(qubit_gate_name, qubit)
            qubit_pulse = next(inst.pulse for _, inst in qubit_sched.instructions
                               if isinstance(inst, pulse.Play))
            for param_name in ['duration', 'sigma']:
                value = getattr(qubit_pulse, param_name)
                calibrations.add_parameter_value(ParameterValue(value), param_name, qubits=[qubit],
                                                 schedule=gate_name)
            for param_name in ['phase_offset', 'amp', 'beta', 'freq']:
                calibrations.add_parameter_value(ParameterValue(0.), param_name, qubits=[qubit],
                                                 schedule=gate_name)

    # Stark delta parameters
    for name in ['x12stark', 'sx12stark']:
        if name not in set(p.name for p in calibrations.parameters.keys()):
            calibrations._register_parameter(Parameter(name), ())

        for qubit in range(backend.num_qubits):
            calibrations.add_parameter_value(ParameterValue(0.), name, qubits=[qubit])


def set_xstark_sxstark_default(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    # Stark delta parameters
    for name in ['xstark', 'sxstark']:
        if name not in set(p.name for p in calibrations.parameters.keys()):
            calibrations._register_parameter(Parameter(name), ())

        for qubit in range(backend.num_qubits):
            calibrations.add_parameter_value(ParameterValue(0.), name, qubits=[qubit])
