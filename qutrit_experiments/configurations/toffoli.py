# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability

from ..data_processing import ReadoutMitigation
from ..experiment_config import BatchExperimentConfig, ExperimentConfig, register_exp, register_post
from ..pulse_library import ModulatedGaussianSquare
from ..util.pulse_area import rabi_freq_per_amp
from .qutrit import (
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
)

logger = logging.getLogger(__name__)


def register_single_qutrit_exp(function):
    @wraps(function)
    def conf_gen(runner):
        return function(runner, runner.program_data['qubits'][1])

    register_exp(conf_gen)

qutrit_functions = [
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
]
for func in qutrit_functions:
    register_single_qutrit_exp(func)

def add_readout_mitigation(gen=None, *, logical_qubits=None):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    if gen is None:
        def wrapper(gen):
            return add_readout_mitigation(gen, logical_qubits=logical_qubits)
        return wrapper

    @wraps(gen)
    def converted_gen(runner):
        config = gen(runner)
        if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
            logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation for %s',
                           gen.__name__)
            return config

        if logical_qubits is not None:
            qubits = tuple(config.physical_qubits[q] for q in logical_qubits)
        else:
            qubits = tuple(config.physical_qubits)

        if (matrix := runner.program_data.get('readout_assignment_matrices', {}).get(qubits)) is None:
            logger.warning('Assignment matrix missing; no readout mitigation for %s',
                           gen.__name__)
            return config

        if (processor := config.analysis_options.get('data_processor')) is None:
            config.analysis_options['data_processor'] = DataProcessor('counts', [
                ReadoutMitigation(matrix),
                Probability(config.analysis_options.get('outcome', '1' * len(qubits)))
            ])
        else:
            processor._nodes.insert(0, ReadoutMitigation(matrix))
        return config

    return converted_gen

@register_exp
def qubits_assignment_error(runner):
    from ..experiments.readout_error import CorrelatedReadoutError
    return ExperimentConfig(
        CorrelatedReadoutError,
        runner.program_data['qubits']
    )

@register_post
def qubits_assignment_error(runner, experiment_data):
    qubits = tuple(experiment_data.metadata['physical_qubits'])
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator', block=False).value
    prog_data = runner.program_data.setdefault('readout_assignment_matrices', {})
    for combination in [qubits[0:1], qubits[1:2], qubits[2:3], qubits[:2], qubits[1:3], qubits]:
        prog_data[combination] = mitigator.assignment_matrix(combination)

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_sizzle_template(runner):
    """Template for SiZZle measurement at a single frequency and amplitude combination.

    Frequency, amplitudes should be set in args.
    """
    from ..experiments.sizzle import SiZZle

    cr_angle = runner.calibrations.get_parameter_value('cr_angle', [control2, target],
                                                       schedule='cr')

    return ExperimentConfig(
        SiZZle,
        runner.program_data['qubits'][1:],
        args={
            'frequency': None,
            'amplitudes': None,
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'control_phase_offset': cr_angle
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_zzramsey(runner):
    from ..experiments.zzramsey import QutritZZRamsey
    return ExperimentConfig(
        QutritZZRamsey,
        runner.program_data['qubits'][1:],
        args={
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6
        },
        run_options={'shots': 4000}
    )

@register_post
def c2t_zzramsey(runner, experiment_data):
    omega_zs = experiment_data.analysis_results('omega_zs', block=False).value
    runner.program_data['c2t_static_omega_zs'] = omega_zs

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_sizzle_frequency_scan(runner):
    from ..experiments.sizzle import SiZZleFrequencyScan
    from ..experiments.zzramsey import QutritZZRamsey

    control2, target = runner.program_data['qubits'][1:]
    c2_props = runner.backend.qubit_properties(control2)
    t_props = runner.backend.qubit_properties(target)

    resonances = {
        'f_ge_c2': c2_props.frequency,
        'f_ef_c2': c2_props.frequency + c2_props.anharmonicity,
        'f_ge_t': t_props.frequency,
        'f_ef_t': t_props.frequency + t_props.anharmonicity
    }
    frequencies = []
    for res_freq in resonances.values():
        frequencies.extend([res_freq - 4.e+6, ])
    for freq in np.linspace(min(resonances.values()) - 1.e+8, max(resonances.values()) + 1.e+8, 20):
        if all(abs(freq - res) > 2.e+6 for res in resonances.values()):
            frequencies.append(freq)

    cr_angle = runner.calibrations.get_parameter_value('cr_angle', [control2, target],
                                                       schedule='cr')
    return ExperimentConfig(
        SiZZleFrequencyScan,
        [control2, target],
        args={
            'frequencies': frequencies,
            'amplitudes': (0.1, 0.1),
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'control_phase_offset': cr_angle
        },
        experiment_options={'frequencies_of_interest': resonances}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_sizzle_amplitude_scan_template(runner):
    from ..experiments.sizzle import SiZZleAmplitudeScan

    control2, target = runner.program_data['qubits'][1:]
    cr_angle = runner.calibrations.get_parameter_value('cr_angle', [control2, target],
                                                       schedule='cr')
    return ExperimentConfig(
        SiZZleAmplitudeScan,
        [control2, target],
        args={
            'frequency': None,
            'amplitudes': None,
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'control_phase_offset': cr_angle
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_hcr_template(runner):
    """CR HT with undefined amplitude."""
    from ..experiments.qutrit_cr_hamiltonian import QutritCRHamiltonianTomography

    control2, target = runner.program_data['qubits'][1:]
    width = Parameter('width')
    cr_amp = Parameter('cr_amp')
    assign_params = {'width': width, 'cr_amp': cr_amp}
    schedule = runner.calibrations.get_schedule('cr', qubits=[control2, target],
                                                assign_params=assign_params)
    return ExperimentConfig(
        QutritCRHamiltonianTomography,
        [control2, target],
        args={
            'schedule': schedule
        },
        analysis_options={
            # Need to set this manually to avoid being overwritten by add_readout_mitigation
            'data_processor': DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        }
    )

@register_exp
@add_readout_mitigation
def t_hx(runner):
    """Target qubit Rx tone Hamiltonian with amplitude set for two cycles in 2048 dt."""
    from ..experiments.hamiltonian_tomography import HamiltonianTomography

    qubit = runner.program_data['qubits'][2]
    width = Parameter('width')
    sigma = 64
    rsr = 2
    duration = width + sigma * rsr * 2
    amp = 2. / 2048 / runner.backend.dt / rabi_freq_per_amp(runner.backend, qubit)
    with pulse.build(name='rx') as sched:
        pulse.play(
            pulse.GaussianSquare(duration=duration, amp=amp, sigma=sigma, width=width),
            runner.backend.drive_channel(qubit)
        )

    return ExperimentConfig(
        HamiltonianTomography,
        [qubit],
        args={
            'schedule': sched,
            'widths': np.linspace(0., 2048., 17)
        },
        analysis_options={
            # Need to set this manually to avoid being overwritten by add_readout_mitigation
            'data_processor': DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        }
    )

@register_post
def t_hx(runner, data):
    components = data.analysis_results('hamiltonian_components', block=False)
    runner.program_data['target_rxtone_hamiltonian'] = components

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_hcr_singlestate_template(runner):
    """CR Hamiltonian tomography with a single control state and an offset Rx tone.

    Control state and cr_amp should be set through cr_rabi_init(state) and
    schedule.assign_parameters(), respectively, in args after the experiment configuration is
    instantiated.
    """
    from ..experiments.hamiltonian_tomography import HamiltonianTomography

    control2, target = runner.program_data['qubits'][1:]
    width = Parameter('width')
    cr_amp = Parameter('cr_amp')
    counter_amp = 2. / 2048 / runner.backend.dt / rabi_freq_per_amp(runner.backend, target)
    assign_params = {'width': width, 'cr_amp': cr_amp, 'counter_amp': counter_amp}
    schedule = runner.calibrations.get_schedule('cr', qubits=[control2, target],
                                                assign_params=assign_params)
    return ExperimentConfig(
        HamiltonianTomography,
        [control2, target],
        args={
            'schedule': schedule,
            'rabi_init': None,
            'measured_logical_qubit': 1
        },
        analysis_options={
            # Need to set this manually to avoid being overwritten by add_readout_mitigation
            'data_processor': DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_hcr_amplitude_scan(runner):
    from ..experiments.qutrit_cr_hamiltonian import QutritCRHamiltonianTomographyScan

    control2, target = runner.program_data['qubits'][1:]
    width = Parameter('width')
    cr_amp = Parameter('cr_amp')
    assign_params = {'width': width, 'cr_amp': cr_amp}
    schedule = runner.calibrations.get_schedule('cr', qubits=[control2, target],
                                                assign_params=assign_params)
    amplitudes = np.linspace(0.1, 0.9, 9)
    poly_orders = ({(ic, ib): [1, 3] for ic in range(3) for ib in [0, 1]}
                   | {(ic, 2): [0, 2] for ic in range(3)})

    return ExperimentConfig(
        QutritCRHamiltonianTomographyScan,
        [control2, target],
        args={
            'schedule': schedule,
            'parameter': 'cr_amp',
            'values': amplitudes
        },
        analysis_options={
            # Need to set this manually to avoid being overwritten by add_readout_mitigation
            'data_processor': DataProcessor('counts', [Probability('1'), BasisExpectationValue()]),
            'poly_orders': poly_orders
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_rcrx_rotary(runner):
    from ..experiments.qutrit_qubit_cx.repeated_cr_rotary import RepeatedCRRotaryAmplitudeCal
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:],
        args={
            'rcr_type': 'x'
        },
        analysis_options={
            'data_processor': DataProcessor('counts', [])
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_rcrx12_rotary(runner):
    from ..experiments.qutrit_qubit_cx.repeated_cr_rotary import RepeatedCRRotaryAmplitudeCal
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:],
        args={
            'rcr_type': 'x12'
        },
        analysis_options={
            'data_processor': DataProcessor('counts', [])
        }
    )
