# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Config generator prototypes for qutrit gate calibrations."""
import numpy as np
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.data_processing import (DataProcessor, DiscriminatorNode, MemoryToCounts,
                                                Probability)

from ..experiment_config import ExperimentConfig


def qutrit_rough_frequency(runner, qubit):
    from ..experiments.rough_frequency import EFRoughFrequencyCal
    return ExperimentConfig(EFRoughFrequencyCal, [qubit])

def qutrit_rough_frequency_post(runner, experiment_data, qubit):
    freq = (runner.calibrations.get_parameter_value('f12', qubit)
            - runner.backend.qubit_properties(qubit).frequency) * runner.backend.dt
    for sched_name in ['x12', 'sx12']:
        for group in ['qutrit_rough_frequency', 'default']:
            BaseUpdater.add_parameter_value(runner.calibrations, experiment_data, freq, 'freq',
                                            schedule=sched_name, group=group)


def qutrit_rough_amplitude(runner, qubit):
    from ..experiments.rough_amplitude import EFRoughXSXAmplitudeCal
    return ExperimentConfig(EFRoughXSXAmplitudeCal, [qubit])

def qutrit_semifine_frequency(runner, qubit):
    from ..experiments.delay_phase_offset import EFRamseyPhaseSweepFrequencyCal
    config = ExperimentConfig(
        EFRamseyPhaseSweepFrequencyCal,
        [qubit]
    )
    return _add_iq_discriminator(config, runner)

def qutrit_erroramp_frequency(runner, qubit):
    from ..experiments.fine_frequency import EFFineFrequency

    delay_duration = runner.calibrations.get_schedule('sx12', qubit).duration

    config = ExperimentConfig(
        EFFineFrequency,
        [qubit],
        args={'delay_duration': delay_duration}
    )
    return _add_iq_discriminator(config, runner)

def qutrit_fine_frequency(runner, qubit):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScanCal

    config = ExperimentConfig(
        EFRamseyFrequencyScanCal,
        [qubit]
    )
    if not runner.data_taking_only:
        config.analysis_options['parallelize'] = 0

    return _add_iq_discriminator(config, runner)

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

    return _add_iq_discriminator(config, runner)

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
    return _add_iq_discriminator(config, runner)

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
    return _add_iq_discriminator(config, runner)

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
    return _add_iq_discriminator(config, runner)

def qutrit_fine_sx_drag(runner, qubit):
    from ..experiments.fine_drag import EFFineSXDragCal

    config = ExperimentConfig(
        EFFineSXDragCal,
        [qubit],
        experiment_options={
            'repetitions': list(range(12))
        }
    )
    return _add_iq_discriminator(config, runner)

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
    return _add_iq_discriminator(config, runner)

def qutrit_x12_stark_shift(runner, qubit):
    from ..experiments.stark_shift_phase import X12StarkShiftPhaseCal

    return ExperimentConfig(
        X12StarkShiftPhaseCal,
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

def qutrit_assignment_error_post(runner, experiment_data, qubit):
    matrix = experiment_data.analysis_results('assignment_matrix').value
    runner.program_data.setdefault('qutrit_assignment_matrix', {})[qubit] = matrix

def qutrit_t1(runner, qubit):
    from ..experiments.ef_t1 import EFT1

    assignment_matrix = runner.program_data['qutrit_assignment_matrix'][qubit]

    return ExperimentConfig(
        EFT1,
        [qubit],
        analysis_options={'assignment_matrix': assignment_matrix}
    )

def _add_iq_discriminator(config, runner):
    qubit = config.physical_qubits[0]
    config.run_options.update({
        'meas_level': MeasLevel.KERNELED,
        'meas_return': MeasReturnType.SINGLE
    })
    config.analysis_options['data_processor'] = DataProcessor('memory', [
        DiscriminatorNode(runner.program_data['iq_discriminator'][qubit]),
        MemoryToCounts(),
        Probability(config.analysis_options.get('outcome', '1'))
    ])
    return config
