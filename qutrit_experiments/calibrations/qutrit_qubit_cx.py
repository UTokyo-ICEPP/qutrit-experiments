"""Functions to generate the Calibrations object for qutrit experiments."""

import logging
from typing import Optional
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

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
    stark_detuning = Parameter('stark_detuning')

    cr_amp = Parameter('cr_amp')
    cr_angle = Parameter('cr_angle')
    cr_stark_amp = Parameter('cr_stark_amp')
    cr_stark_phase = Parameter('cr_stark_phase')
    cr_full_amp = cr_amp + cr_stark_amp

    counter_amp = Parameter('counter_amp')
    counter_angle = Parameter('counter_angle')
    counter_stark_amp = Parameter('counter_stark_amp')
    counter_full_amp = counter_amp + counter_stark_amp

    cr_pulse = ModulatedGaussianSquare(duration=duration, amp=cr_full_amp,
                                       sigma=sigma, freq=(0., stark_detuning),
                                       fractions=(cr_amp, cr_stark_amp), width=width,
                                       angle=cr_angle, risefall_sigma_ratio=rsr,
                                       phases=(cr_stark_phase,), name='CR')
    counter_pulse = ModulatedGaussianSquare(duration=duration, amp=counter_full_amp,
                                            sigma=sigma, freq=(0., stark_detuning),
                                            fractions=(counter_amp, counter_stark_amp), width=width,
                                            angle=counter_angle, risefall_sigma_ratio=rsr,
                                            name='Counter')

    control_channel = pulse.ControlChannel(Parameter('ch0.1'))
    target_channel = pulse.DriveChannel(Parameter('ch1'))

    with pulse.build(name='cr', default_alignment='left') as sched:
        pulse.play(cr_pulse, control_channel, name='CR')
        pulse.play(counter_pulse, target_channel, name='Counter')
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits, control_channel in backend.control_channels.items():
        ecr_sched = get_default_ecr_schedule(backend, qubits)
        control_instructions = [inst for _, inst in ecr_sched.instructions
                                if isinstance(inst, pulse.Play) and
                                   inst.channel == control_channel]
        target_instructions = [inst for _, inst in ecr_sched.instructions
                               if isinstance(inst, pulse.Play) and
                                  inst.channel == backend.drive_channel(qubits[1])]
        try:
            default_cr = next(inst.pulse for inst in control_instructions
                              if inst.name.startswith('CR90m'))
            default_rotary = next(inst.pulse for inst in target_instructions
                                  if inst.name.startswith('CR90m'))
        except StopIteration: # Direct CX
            default_cr = next(inst.pulse for inst in control_instructions
                              if inst.name.startswith('CX'))
            default_rotary = None

        param_defaults = [
            ('rsr', (default_cr.duration - default_cr.width) / 2. / default_cr.sigma),
            ('sigma', default_cr.sigma),
            ('width', 0.),
            ('margin', 0.),
            ('stark_detuning', 0.),
            ('cr_amp', 0.),
            ('cr_angle', default_cr.angle),
            ('cr_stark_amp', 0.),
            ('cr_stark_phase', 0.),
            ('counter_amp', 0.),
            ('counter_angle', 0. if default_rotary is None else default_rotary.angle),
            ('counter_stark_amp', 0.)
        ]
        for pname, value in param_defaults:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=qubits,
                                             schedule='cr')
