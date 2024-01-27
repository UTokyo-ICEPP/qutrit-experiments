"""Functions to generate the Calibrations object for qutrit experiments."""

import logging
from typing import Optional
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management import Calibrations, ParameterValue
from qiskit_experiments.exceptions import CalibrationError

from .util import get_default_ecr_schedule
from ..pulse_library import ModulatedGaussianSquare

logger = logging.getLogger(__name__)


def make_qutrit_qubit_cx_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)

    add_qutrit_qubit_cr(backend, calibrations)
    add_offset_rx(backend, calibrations)

    return calibrations


def add_qutrit_qubit_cr(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    """Define CR and counter tones with ModulatedGaussianSquare potentiallly with an additional
    off-resonant component."""
    rsr = Parameter('rsr')
    sigma = Parameter('sigma')
    width = Parameter('width')
    margin = Parameter('margin')
    duration = sigma * rsr * 2 + width + margin
    stark_frequency = Parameter('stark_frequency')
    stark_detuning = (stark_frequency - Parameter('target_frequency')) * backend.dt

    cr_amp = Parameter('cr_amp')
    cr_base_angle = Parameter('cr_base_angle')
    cr_sign_angle = Parameter('cr_sign_angle')
    cr_stark_amp = Parameter('cr_stark_amp')
    cr_stark_base_phase = Parameter('cr_stark_base_phase')
    cr_stark_sign_phase = Parameter('cr_stark_sign_phase')
    cr_full_amp = cr_amp + cr_stark_amp

    counter_amp = Parameter('counter_amp')
    counter_base_angle = Parameter('counter_base_angle')
    counter_sign_angle = Parameter('counter_sign_angle')
    counter_stark_amp = Parameter('counter_stark_amp')
    counter_full_amp = counter_amp + counter_stark_amp

    cr_pulse = ModulatedGaussianSquare(duration=duration, amp=cr_full_amp,
                                       sigma=sigma, freq=(0., stark_detuning),
                                       fractions=(cr_amp, cr_stark_amp), width=width,
                                       angle=cr_base_angle + cr_sign_angle, risefall_sigma_ratio=rsr,
                                       phases=(cr_stark_base_phase + cr_stark_sign_phase,),
                                       name='CR')
    counter_pulse = ModulatedGaussianSquare(duration=duration, amp=counter_full_amp,
                                            sigma=sigma, freq=(0., stark_detuning),
                                            fractions=(counter_amp, counter_stark_amp), width=width,
                                            angle=counter_base_angle + counter_sign_angle,
                                            risefall_sigma_ratio=rsr, name='Counter')

    with pulse.build(name='cr', default_alignment='left') as sched:
        pulse.play(cr_pulse, pulse.ControlChannel(Parameter('ch0.1')), name='CR')
        pulse.play(counter_pulse, pulse.DriveChannel(Parameter('ch1')), name='Counter')
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits, control_channels in backend.control_channels.items():
        control_channel = control_channels[0]
        target_channel = backend.drive_channel(qubits[1])
        target_frequency = backend.qubit_properties(qubits[1]).frequency
        ecr_sched = get_default_ecr_schedule(backend, qubits)
        control_instructions = [inst for _, inst in ecr_sched.instructions
                                if isinstance(inst, pulse.Play) and
                                   inst.channel == control_channel]
        if not control_instructions:
            # Reverse CR
            continue
        target_instructions = [inst for _, inst in ecr_sched.instructions
                               if isinstance(inst, pulse.Play) and
                                  inst.channel == target_channel]
        try:
            default_cr = next(inst.pulse for inst in control_instructions
                              if inst.name.startswith('CR90m'))
            default_rotary = next(inst.pulse for inst in target_instructions
                                  if inst.name.startswith('CR90m'))
        except StopIteration: # Direct CX
            try:
                default_cr = next(inst.pulse for inst in control_instructions
                                  if inst.name.startswith('CX'))
            except StopIteration as exc:
                raise CalibrationError(f'No default CR instruction for qubits {qubits}') from exc
            default_rotary = None

        param_defaults = [
            ('rsr', (default_cr.duration - default_cr.width) / 2. / default_cr.sigma),
            ('sigma', default_cr.sigma),
            ('width', 0.),
            ('margin', 0.),
            ('stark_frequency', target_frequency),
            ('target_frequency', target_frequency),
            ('cr_amp', 0.),
            ('cr_base_angle', default_cr.angle),
            ('cr_sign_angle', 0.),
            ('cr_stark_amp', 0.),
            ('cr_stark_base_phase', 0.),
            ('cr_stark_sign_phase', 0.),
            ('counter_amp', 0.),
            ('counter_base_angle', 0. if default_rotary is None else default_rotary.angle),
            ('counter_sign_angle', 0.),
            ('counter_stark_amp', 0.)
        ]
        for pname, value in param_defaults:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=qubits,
                                             schedule='cr')

def add_offset_rx(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    """Add the schedule for offset Rx on the target qubit."""
    duration = Parameter('duration')
    amp = Parameter('amp')
    sigma = Parameter('sigma')
    base_angle = Parameter('base_angle')
    sign_angle = Parameter('sign_angle')

    rx_pulse = pulse.Gaussian(duration=duration, amp=amp, sigma=sigma, angle=base_angle + sign_angle,
                              name='Rx')

    with pulse.build(name='offset_rx') as sched:
        pulse.play(rx_pulse, pulse.DriveChannel(Parameter('ch0')), name='Rx')
    calibrations.add_schedule(sched, num_qubits=1)

    for _, qubit in backend.control_channels.keys():
        x_sched = backend.defaults().instruction_schedule_map.get('x', qubit)
        x_pulse = next(inst.pulse for _, inst in x_sched.instructions
                       if isinstance(inst, pulse.Play))
        param_defaults = [
            ('duration', x_sched.duration),
            ('amp', 0.),
            ('sigma', x_sched.duration / 4),
            ('base_angle', x_pulse.angle),
            ('sign_angle', 0.)
        ]
        for pname, value in param_defaults:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='offset_rx')
