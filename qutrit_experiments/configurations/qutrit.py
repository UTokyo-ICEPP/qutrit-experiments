# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Config generator prototypes for qutrit gate calibrations."""
import logging
import numpy as np
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.data_processing import (DataProcessor, DiscriminatorNode, MemoryToCounts,
                                                Probability)
from qiskit_experiments.visualization import MplDrawer, IQPlotter

from ..calibrations import get_qutrit_pulse_gate
from ..data_processing import LinearDiscriminator
from ..experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def qutrit_rough_frequency(runner, qubit):
    """EF frequency measurement based on spectroscopy."""
    from ..experiments.rough_frequency import EFRoughFrequencyCal
    return ExperimentConfig(EFRoughFrequencyCal, [qubit])

def qutrit_rough_amplitude(runner, qubit):
    """X12 and SX12 amplitude determination from Rabi oscillation."""
    from ..experiments.rough_amplitude import EFRoughXSXAmplitudeCal
    return ExperimentConfig(
        EFRoughXSXAmplitudeCal,
        [qubit],
        experiment_options={'discrimination_basis': '02'},
        restless=True,
        # rabi rate = "freq" of oscillation analysis fit. Must be greater than 0.5 to be able to
        # make a pi pulse within amp < 1
        calibration_criterion=lambda data: data.analysis_results('rabi_rate_12').value.n > 0.5
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

def qutrit_semifine_frequency(runner, qubit):
    from ..experiments.delay_phase_offset import EFRamseyPhaseSweepFrequencyCal
    config = ExperimentConfig(
        EFRamseyPhaseSweepFrequencyCal,
        [qubit],
        analysis_options={'common_amp': False}
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_fine_frequency(runner, qubit):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScanCal

    config = ExperimentConfig(
        EFRamseyFrequencyScanCal,
        [qubit],
        analysis_options={'common_amp': False}
    )
    if not runner.data_taking_only:
        config.analysis_options['parallelize'] = 0

    #return _add_iq_discriminator(config, runner)
    return config

def nocal_qutrit_fine_frequency(runner, qubit):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScan

    config = ExperimentConfig(
        EFRamseyFrequencyScan,
        [qubit],
        args={
            'frequencies': (np.linspace(-5.e+5, 5.e+5, 6)
                            + runner.calibrations.get_parameter_value('f12', qubit))
        }
    )
    if not runner.data_taking_only:
        config.analysis_options['parallelize'] = 0

    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_rough_x_drag(runner, qubit):
    from ..experiments.rough_drag import EFRoughDragCal

    config = ExperimentConfig(
        EFRoughDragCal,
        [qubit],
        args={
            'schedule_name': 'x12',
            'betas': np.linspace(-10., 10., 10)
        }
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_rough_sx_drag(runner, qubit):
    from ..experiments.rough_drag import EFRoughDragCal

    config = ExperimentConfig(
        EFRoughDragCal,
        [qubit],
        args={
            'schedule_name': 'sx12',
            'betas': np.linspace(-20., 20., 10)
        }
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_fine_sx_amplitude(runner, qubit):
    from ..experiments.fine_amplitude import EFFineSXAmplitudeCal

    # qutrit T1 is short - shouldn't go too far with repetitions
    config = ExperimentConfig(
        EFFineSXAmplitudeCal,
        [qubit],
        experiment_options={
            'repetitions': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15],
            'normalization': False,
        }
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_fine_sx_drag(runner, qubit):
    from ..experiments.fine_drag import EFFineSXDragCal

    config = ExperimentConfig(
        EFFineSXDragCal,
        [qubit],
        experiment_options={
            'repetitions': list(range(12))
        }
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_fine_x_amplitude(runner, qubit):
    from ..experiments.fine_amplitude import EFFineXAmplitudeCal

    # qutrit T1 is short - shouldn't go too far with repetitions
    config = ExperimentConfig(
        EFFineXAmplitudeCal,
        [qubit],
        experiment_options={
            'normalization': False,
        }
    )
    return _add_iq_discriminator(config, runner)

def qutrit_fine_x_drag(runner, qubit):
    from ..experiments.fine_drag import EFFineXDragCal

    config = ExperimentConfig(
        EFFineXDragCal,
        [qubit],
        experiment_options={
            'repetitions': list(range(12)),
        }
    )
    #return _add_iq_discriminator(config, runner)
    return config

def qutrit_x12_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import X12StarkShiftPhaseCal

    return ExperimentConfig(
        X12StarkShiftPhaseCal,
        [qubit]
    )

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

def qutrit_sx_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import SXStarkShiftPhaseCal

    return ExperimentConfig(
        SXStarkShiftPhaseCal,
        [qubit]
    )

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
    matrix = experiment_data.analysis_results('assignment_matrix').value
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

def qutrit_x12_irb(runner, qubit):
    from ..experiments.qutrit_rb import QutritInterleavedRB
    from ..gates import X12Gate

    return ExperimentConfig(
        QutritInterleavedRB,
        [qubit],
        args={'interleaved_gate': X12Gate}
    )

def _add_iq_discriminator(config, runner):
    qubit = config.physical_qubits[0]

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
