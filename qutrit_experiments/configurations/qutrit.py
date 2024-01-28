# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Config generator prototypes for qutrit gate calibrations."""
from functools import wraps
import logging
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.data_processing import (DataProcessor, DiscriminatorNode, MemoryToCounts,
                                                Probability)
from qiskit_experiments.visualization import MplDrawer, IQPlotter

from ..calibrations import get_qutrit_pulse_gate
from ..data_processing import LinearDiscriminator, ReadoutMitigation
from ..experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def add_iq_discriminator(gen):
    """Decorator to convert an experiment meas_level to kerneled and add the IQ-plane discriminator
    node to the DataProcessor."""
    @wraps(gen)
    def converted_gen(runner, qubit):
        config = gen(runner, qubit)
        if (discriminator := runner.program_data.get('iq_discriminator', {}).get(qubit)) is None:
            logger.warning('IQ discriminator is missing; proceeding with meas_level=2')
            return config

        config.run_options.update({
            'meas_level': MeasLevel.KERNELED,
            'meas_return': MeasReturnType.SINGLE
        })
        config.analysis_options['data_processor'] = DataProcessor('memory', [
            DiscriminatorNode(discriminator),
            MemoryToCounts(),
            Probability(config.analysis_options.get('outcome', '1'))
        ])
        return config

    return converted_gen

def add_readout_mitigation(gen):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    @wraps(gen)
    def converted_gen(runner, qubit):
        config = gen(runner, qubit)
        if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
            logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation for qubit %d',
                           qubit)
            return config
        if (matrix := runner.program_data.get('qubit_assignment_matrix', {}).get(qubit)) is None:
            logger.warning('Assignment matrix missing; no readout mitigation for qubit %d',
                           qubit)
            return config

        if (processor := config.analysis_options.get('data_processor')) is None:
            config.analysis_options['data_processor'] = DataProcessor('counts', [
                ReadoutMitigation(matrix),
                Probability(config.analysis_options.get('outcome', '1'))
            ])
        else:
            probability_pos = next(i for i, node in enumerate(processor._nodes)
                                   if isinstance(node, Probability))
            processor._nodes.insert(probability_pos, ReadoutMitigation(matrix))
        return config

    return converted_gen


def qutrit_rough_frequency(runner, qubit):
    """EF frequency measurement based on spectroscopy."""
    from ..experiments.rough_frequency import EFRoughFrequencyCal
    sx_sched = runner.backend.defaults().instruction_schedule_map.get('sx', qubit)
    sx_pulse = next(inst.pulse for _, inst in sx_sched.instructions if isinstance(inst, pulse.Play))
    sx_amp = sx_pulse.amp
    sx_duration = sx_pulse.duration
    sx_sigma = sx_pulse.sigma
    # aim for pi/2 rotation at the resonance with stretched pulse
    # factor sqrt(2) for ef transition amp
    factor = 4.
    while (amp := sx_amp / factor / np.sqrt(2.)) < 0.005:
        factor *= 0.9
    return ExperimentConfig(
        EFRoughFrequencyCal,
        [qubit],
        experiment_options={
            'amp': amp,
            'duration': sx_duration * factor * runner.backend.dt,
            'sigma': sx_sigma * factor * runner.backend.dt
        }
    )

def qutrit_rough_amplitude(runner, qubit):
    """X12 and SX12 amplitude determination from Rabi oscillation."""
    from ..experiments.rough_amplitude import EFRoughXSXAmplitudeCal
    def calibration_criterion(data):
        return data.analysis_results('rabi_rate_12', block=False).value.n > 0.5
    return ExperimentConfig(
        EFRoughXSXAmplitudeCal,
        [qubit],
        experiment_options={'discrimination_basis': '02'},
        restless=True,
        # rabi rate = "freq" of oscillation analysis fit. Must be greater than 0.5 to be able to
        # make a pi pulse within amp < 1
        calibration_criterion=calibration_criterion
    )

def qutrit_discriminator(runner, qubit):
    """0-2 discriminator determination using results from a Rabi experiment."""
    from ..experiments.rabi import EFRabi
    schedule = get_qutrit_pulse_gate('x12', qubit, runner.backend, runner.calibrations,
                                     assign_params={'amp': Parameter("amp")})
    return ExperimentConfig(
        EFRabi,
        [qubit],
        args={'schedule': schedule}
    )

def qutrit_discriminator_post(runner, experiment_data):
    from ..util.ef_discriminator import ef_discriminator_analysis

    amps = np.array([d['metadata']['xval'] for d in experiment_data.data()])
    theta, dist = ef_discriminator_analysis(experiment_data, np.argmin(np.abs(amps)))
    discriminator = LinearDiscriminator(theta, dist)
    runner.program_data.setdefault('iq_discriminator', {})[runner.program_data['qubit']] = discriminator
    runner.save_program_data('iq_discriminator')

    if False:
        for iamp, datum in enumerate(experiment_data.data()):
            amplitude = datum['metadata']['amplitude']
            plotter = IQPlotter(MplDrawer())
            plotter.set_series_data('0', points=np.squeeze(datum['memory']))
            plotter.set_figure_options(series_params={'0': {'label': f'amp={amplitude}'}})
            plotter.set_supplementary_data(discriminator=discriminator)
            experiment_data.add_figures(plotter.figure(), f'iq_{iamp}')

def qubit_assignment_error(runner, qubit):
    from ..experiments.readout_error import CorrelatedReadoutError
    return ExperimentConfig(
        CorrelatedReadoutError,
        [qubit]
    )

def qubit_assignment_error_post(runner, experiment_data):
    qubit = experiment_data.metadata['physical_qubits'][0]
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator', block=False).value
    runner.program_data.setdefault('qubit_assignment_matrix', {})[qubit] = \
        mitigator.assignment_matrix([qubit])

@add_readout_mitigation
def qutrit_semifine_frequency(runner, qubit):
    from ..experiments.delay_phase_offset import EFRamseyPhaseSweepFrequencyCal
    return ExperimentConfig(
        EFRamseyPhaseSweepFrequencyCal,
        [qubit],
        analysis_options={'common_amp': False}
    )

@add_readout_mitigation
def qutrit_fine_frequency(runner, qubit):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScanCal
    return ExperimentConfig(
        EFRamseyFrequencyScanCal,
        [qubit],
        analysis_options={'common_amp': False}
    )

@add_readout_mitigation
def nocal_qutrit_fine_frequency(runner, qubit):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScan
    return ExperimentConfig(
        EFRamseyFrequencyScan,
        [qubit],
        args={
            'frequencies': (np.linspace(-5.e+5, 5.e+5, 6)
                            + runner.calibrations.get_parameter_value('f12', qubit))
        }
    )

@add_readout_mitigation
def qutrit_rough_x_drag(runner, qubit):
    from ..experiments.rough_drag import EFRoughDragCal
    return ExperimentConfig(
        EFRoughDragCal,
        [qubit],
        args={
            'schedule_name': 'x12',
            'betas': np.linspace(-10., 10., 10)
        }
    )

@add_readout_mitigation
def qutrit_rough_sx_drag(runner, qubit):
    from ..experiments.rough_drag import EFRoughDragCal
    return ExperimentConfig(
        EFRoughDragCal,
        [qubit],
        args={
            'schedule_name': 'sx12',
            'betas': np.linspace(-20., 20., 10)
        }
    )

@add_readout_mitigation
def qutrit_fine_sx_amplitude(runner, qubit):
    from ..experiments.fine_amplitude import EFFineSXAmplitudeCal

    def calibration_criterion(data):
        target_angle = data.metadata["target_angle"]
        prev_amp = data.metadata["cal_param_value"]
        if isinstance(prev_amp, list) and len(prev_amp) == 2:
            prev_amp = prev_amp[0] + 1.0j * prev_amp[1]
        d_theta = data.analysis_results("d_theta", block=False).value.n
        return abs(prev_amp * target_angle / (target_angle + d_theta)) < 1.

    # qutrit T1 is short - shouldn't go too far with repetitions
    return ExperimentConfig(
        EFFineSXAmplitudeCal,
        [qubit],
        experiment_options={
            'repetitions': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15],
            'normalization': False,
        },
        calibration_criterion=calibration_criterion
    )

@add_readout_mitigation
def qutrit_fine_sx_drag(runner, qubit):
    from ..experiments.fine_drag import EFFineSXDragCal
    return ExperimentConfig(
        EFFineSXDragCal,
        [qubit],
        experiment_options={
            'repetitions': list(range(12))
        }
    )

@add_readout_mitigation
def qutrit_fine_x_amplitude(runner, qubit):
    from ..experiments.fine_amplitude import EFFineXAmplitudeCal

    def calibration_criterion(data):
        target_angle = data.metadata["target_angle"]
        prev_amp = data.metadata["cal_param_value"]
        if isinstance(prev_amp, list) and len(prev_amp) == 2:
            prev_amp = prev_amp[0] + 1.0j * prev_amp[1]
        d_theta = data.analysis_results("d_theta", block=False).value.n
        return abs(prev_amp * target_angle / (target_angle + d_theta)) < 1.

    # qutrit T1 is short - shouldn't go too far with repetitions
    return ExperimentConfig(
        EFFineXAmplitudeCal,
        [qubit],
        experiment_options={
            'normalization': False,
        },
        calibration_criterion=calibration_criterion
    )

@add_readout_mitigation
def qutrit_fine_x_drag(runner, qubit):
    from ..experiments.fine_drag import EFFineXDragCal
    return ExperimentConfig(
        EFFineXDragCal,
        [qubit],
        experiment_options={
            'repetitions': list(range(12)),
        }
    )

@add_readout_mitigation
def qutrit_x12_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import X12StarkShiftPhaseCal
    return ExperimentConfig(
        X12StarkShiftPhaseCal,
        [qubit]
    )

@add_readout_mitigation
def qutrit_sx12_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import SX12StarkShiftPhaseCal
    return ExperimentConfig(
        SX12StarkShiftPhaseCal,
        [qubit]
    )

def qutrit_x_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import XStarkShiftPhaseCal
    return ExperimentConfig(
        XStarkShiftPhaseCal,
        [qubit]
    )

@add_readout_mitigation
def qutrit_sx_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import SXStarkShiftPhaseCal
    return ExperimentConfig(
        SXStarkShiftPhaseCal,
        [qubit]
    )

@add_readout_mitigation
def qutrit_rotary_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import RotaryStarkShiftPhaseCal
    return ExperimentConfig(
        RotaryStarkShiftPhaseCal,
        [qubit]
    )

def qutrit_assignment_error(runner, qubit):
    from ..experiments.readout_error import MCMLocalReadoutError
    return ExperimentConfig(
        MCMLocalReadoutError,
        [qubit],
        run_options={'shots': 10000}
    )

def qutrit_assignment_error_post(runner, experiment_data):
    qubit = experiment_data.metadata['physical_qubits'][0]
    matrix = experiment_data.analysis_results('assignment_matrix', block=False).value
    runner.program_data.setdefault('qutrit_assignment_matrix', {})[qubit] = matrix

def qutrit_t1(runner, qubit):
    from ..experiments.ef_t1 import EFT1
    assignment_matrix = runner.program_data['qutrit_assignment_matrix'][qubit]
    return ExperimentConfig(
        EFT1,
        [qubit],
        run_options={'rep_delay': runner.backend.configuration().rep_delay_range[1]},
        analysis_options={'assignment_matrix': assignment_matrix}
    )

@add_readout_mitigation
def qutrit_x12_irb(runner, qubit):
    from ..experiments.qutrit_rb import QutritInterleavedRB
    from ..gates import X12Gate
    return ExperimentConfig(
        QutritInterleavedRB,
        [qubit],
        args={'interleaved_gate': X12Gate},
        analysis_options={'outcome': '0'} # Needed because add_readout_mitigation sets this to '1'
    )
