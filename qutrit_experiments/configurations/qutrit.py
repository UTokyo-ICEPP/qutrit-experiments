"""Config generator prototypes for qutrit gate calibrations."""
import numpy as np

from ..experiment_config import ExperimentConfig, register_post


def qutrit_rough_frequency(runner):
    from ..experiments.rough_frequency import EFRoughFrequencyCal
    return ExperimentConfig(
        EFRoughFrequencyCal
    )

def qutrit_rough_amplitude(runner):
    from ..experiments.ef_discriminator import EFRoughXSXAmplitudeAndDiscriminatorCal
    return ExperimentConfig(
        EFRoughXSXAmplitudeAndDiscriminatorCal,
        args={'fine_boundary_optimization': True}
    )

def qutrit_rough_amplitude_post(runner):
    boundary = {pname: runner.calibrations.get_parameter_value(pname,
                                                               runner.program_data['qubits'][1],
                                                               schedule='iq_discriminator')
                for pname in ['theta', 'dist']}
    runner.program_data['iq_discriminators'] = [boundary]

def qutrit_semifine_frequency(runner):
    from ..experiments.delay_phase_offset import EFRamseyPhaseSweepFrequencyCal
    return ExperimentConfig(
        EFRamseyPhaseSweepFrequencyCal,
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']}
    )

def qutrit_erroramp_frequency(runner):
    from ..experiments.fine_frequency import EFFineFrequency

    qubits = tuple(runner.program_data['qubits'][1:2])
    delay_duration = runner.calibrations.get_schedule('sx12', qubits).duration

    return ExperimentConfig(
        EFFineFrequency,
        qubits,
        args={'delay_duration': delay_duration},
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']}
    )

def qutrit_fine_frequency(runner):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScanCal

    config = ExperimentConfig(
        EFRamseyFrequencyScanCal,
        runner.program_data['qubits'][1:2],
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']}
    )
    if not runner.data_taking_only:
        config.analysis_options['parallelize'] = 0

    return config

def nocal_qutrit_fine_frequency(runner):
    from ..experiments.fine_frequency_phase import EFRamseyFrequencyScanCal

    config = ExperimentConfig(
        EFRamseyFrequencyScanCal,
        runner.program_data['qubits'][1:2],
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']},
        calibration=False
    )
    if not runner.data_taking_only:
        config.analysis_options['parallelize'] = 0

    return config

def qutrit_rough_x_drag(runner):
    from ..experiments.rough_drag import EFRoughDragCal

    return ExperimentConfig(
        EFRoughDragCal,
        runner.program_data['qubits'][1:2],
        args={
            'schedule_name': 'x12',
            'betas': np.linspace(-10., 10., 10)
        },
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']}
    )

def qutrit_rough_sx_drag(runner):
    from ..experiments.rough_drag import EFRoughDragCal

    return ExperimentConfig(
        EFRoughDragCal,
        runner.program_data['qubits'][1:2],
        args={
            'schedule_name': 'sx12',
            'betas': np.linspace(-20., 20., 10)
        },
        experiment_options={'iq_discriminators': runner.program_data['iq_discriminators']}
    )

def qutrit_fine_sx_amplitude(runner):
    from ..experiments.fine_amplitude import EFFineSXAmplitudeCal

    # qutrit T1 is short - shouldn't go too far with repetitions
    return ExperimentConfig(
        EFFineSXAmplitudeCal,
        runner.program_data['qubits'][1:2],
        experiment_options={
            'repetitions': [0, 1, 2, 3, 5, 7, 9, 11, 13, 15],
            'normalization': False,
            'iq_discriminators': runner.program_data['iq_discriminators']
        }
    )

def qutrit_fine_sx_drag(runner):
    from ..experiments.fine_drag import EFFineSXDragCal

    return ExperimentConfig(
        EFFineSXDragCal,
        runner.program_data['qubits'][1:2],
        experiment_options={
            'repetitions': list(range(12)),
            'iq_discriminators': runner.program_data['iq_discriminators']
        }
    )

def qutrit_fine_x_amplitude(runner):
    from ..experiments.fine_amplitude import EFFineXAmplitudeCal

    # qutrit T1 is short - shouldn't go too far with repetitions
    return ExperimentConfig(
        EFFineXAmplitudeCal,
        runner.program_data['qubits'][1:2],
        experiment_options={
            'normalization': False,
            'iq_discriminators': runner.program_data['iq_discriminators']
        }
    )

def qutrit_fine_x_drag(runner):
    from ..experiments.fine_drag import EFFineXDragCal

    return ExperimentConfig(
        EFFineXDragCal,
        runner.program_data['qubits'][1:2],
        experiment_options={
            'repetitions': list(range(12)),
            'iq_discriminators': runner.program_data['iq_discriminators']
        }
    )

def qutrit_x12_stark_shift(runner):
    from ..experiments.stark_shift_phase import X12StarkShiftPhaseCal

    return ExperimentConfig(
        X12StarkShiftPhaseCal,
        runner.program_data['qubits'][1:2]
    )

def qutrit_x_stark_shift(runner):
    from ..experiments.stark_shift_phase import XStarkShiftPhaseCal

    return ExperimentConfig(
        XStarkShiftPhaseCal,
        runner.program_data['qubits'][1:2]
    )

def qutrit_sx_stark_shift(runner):
    from ..experiments.stark_shift_phase import SXStarkShiftPhaseCal

    return ExperimentConfig(
        SXStarkShiftPhaseCal,
        runner.program_data['qubits'][1:2]
    )

def qutrit_rotary_stark_shift(runner):
    from ..experiments.stark_shift_phase import RotaryStarkShiftPhaseCal

    return ExperimentConfig(
        RotaryStarkShiftPhaseCal,
        runner.program_data['qubits'][1:2]
    )

def qutrit_assignment_error(runner):
    from ..experiments.readout_error import MCMLocalReadoutError

    return ExperimentConfig(
        MCMLocalReadoutError,
        runner.program_data['qubits'][1:2],
        run_options={'shots': 10000}
    )

def qutrit_assignment_error(runner):
    data = runner.program_data['experiment_data']['qutrit_assignment_error']
    runner.program_data['qutrit_assignment_matrix'] = data.analysis_results('assignment_matrix').value

def qutrit_relaxation_time(runner):
    from ..experiments.ef_t1 import EFT1

    assignment_matrix = runner.program_data['qutrit_assignment_matrix']

    return ExperimentConfig(
        EFT1,
        runner.program_data['qubits'][1:2],
        analysis_options={'assignment_matrix': assignment_matrix}
    )
