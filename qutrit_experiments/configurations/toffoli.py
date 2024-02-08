# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework import BackendTiming

from ..data_processing import ReadoutMitigation
from ..experiment_config import BatchExperimentConfig, ExperimentConfig, register_exp, register_post
from ..pulse_library import ModulatedGaussianSquare
from ..util.pulse_area import rabi_freq_per_amp, grounded_gauss_area
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

def add_readout_mitigation(gen=None, *, logical_qubits=None, expval=False):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    if gen is None:
        def wrapper(gen):
            return add_readout_mitigation(gen, logical_qubits=logical_qubits, expval=expval)
        return wrapper

    @wraps(gen)
    def converted_gen(runner, *args, **kwargs):
        config = gen(runner, *args, **kwargs)
        configure_readout_mitigation(runner, config, logical_qubits=logical_qubits, expval=expval)
        return config

    return converted_gen

def configure_readout_mitigation(runner, config, logical_qubits=None, expval=False):
    if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
        logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation for %s',
                       config.exp_type)
        return

    if logical_qubits is not None:
        qubits = tuple(config.physical_qubits[q] for q in logical_qubits)
    else:
        qubits = tuple(config.physical_qubits)

    if (matrix := runner.program_data.get('readout_assignment_matrices', {}).get(qubits)) is None:
        logger.warning('Assignment matrix missing; no readout mitigation for %s',
                       config.exp_type)
        return

    if (processor := config.analysis_options.get('data_processor')) is None:
        nodes = [
            ReadoutMitigation(matrix),
            Probability(config.analysis_options.get('outcome', '1' * len(qubits)))
        ]
        if expval:
            nodes.append(BasisExpectationValue())
        config.analysis_options['data_processor'] = DataProcessor('counts', nodes)
    else:
        processor._nodes.insert(0, ReadoutMitigation(matrix))

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

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_cr_rough_width(runner):
    """Few-sample CR HT just to measure ωx to find a rough estimate for the CR width in CRCR."""
    from ..experiments.qutrit_qubit_cx.cr_rough_width import CycledRepeatedCRRoughWidthCal
    return ExperimentConfig(
        CycledRepeatedCRRoughWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'widths': np.linspace(0., 4096., 9)
        }
    )

@add_readout_mitigation(logical_qubits=[1])
def c2t_sizzle_template(runner, exp_type):
    """Template for SiZZle measurement at a single frequency and amplitude combination.

    Frequency, amplitudes should be set in args.
    """
    from ..experiments.qutrit_qubit.sizzle import SiZZle
    qubits = runner.program_data['qubits'][1:]

    cr_base_angle = runner.calibrations.get_parameter_value('cr_base_angle', qubits, schedule='cr')
    counter_base_angle = runner.calibrations.get_parameter_value('counter_base_angle', qubits,
                                                                 schedule='cr')

    return ExperimentConfig(
        SiZZle,
        qubits,
        args={
            'frequency': None,
            'amplitudes': None,
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'angles': (cr_base_angle, counter_base_angle)
        },
        exp_type=exp_type
    )

@add_readout_mitigation(logical_qubits=[1])
@register_exp
def c2t_zzramsey(runner):
    from ..experiments.qutrit_qubit.zzramsey import QutritZZRamsey
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

@add_readout_mitigation(logical_qubits=[1])
@register_exp
def c2t_sizzle_frequency_scan(runner):
    from ..experiments.qutrit_qubit.sizzle import SiZZleFrequencyScan

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

    cr_base_angle = runner.calibrations.get_parameter_value('cr_base_angle', [control2, target],
                                                            schedule='cr')
    counter_base_angle = runner.calibrations.get_parameter_value('counter_base_angle',
                                                                 [control2, target], schedule='cr')
    return ExperimentConfig(
        SiZZleFrequencyScan,
        [control2, target],
        args={
            'frequencies': frequencies,
            'amplitudes': (0.1, 0.1),
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'angles': (cr_base_angle, counter_base_angle)
        },
        experiment_options={'frequencies_of_interest': resonances}
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_hcr_template(runner, exp_type):
    """CR HT with undefined amplitude."""
    from ..experiments.qutrit_qubit.qutrit_cr_hamiltonian import QutritCRHamiltonianTomography

    control2, target = runner.program_data['qubits'][1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_frequency': runner.backend.qubit_properties(target).frequency,
        'cr_amp': Parameter('cr_amp'),
        'cr_sign_angle': 0.,
        'cr_stark_amp': 0.,
        'counter_amp': 0.,
        'counter_stark_amp': 0.
    }
    schedule = runner.calibrations.get_schedule('cr', qubits=[control2, target],
                                                assign_params=assign_params)
    return ExperimentConfig(
        QutritCRHamiltonianTomography,
        [control2, target],
        args={
            'schedule': schedule
        },
        exp_type=exp_type
    )

@add_readout_mitigation(expval=True)
@register_exp
def t_hx(runner):
    """Target qubit Rx tone Hamiltonian with amplitude set for two cycles in 2048 dt."""
    from ..experiments.hamiltonian_tomography import HamiltonianTomography

    qubit = runner.program_data['qubits'][2]
    width = Parameter('width')
    sigma = 64
    rsr = 2
    duration = width + sigma * rsr * 2
    amp = 2. / 2048 / runner.backend.dt / rabi_freq_per_amp(runner.backend, qubit)
    angle = runner.calibrations.get_parameter_value('counter_base_angle',
                                                    runner.program_data['qubits'][1:],
                                                    schedule='cr')
    with pulse.build(name='rx') as sched:
        pulse.play(
            pulse.GaussianSquare(duration=duration, amp=amp, sigma=sigma, width=width, angle=angle),
            runner.backend.drive_channel(qubit)
        )

    return ExperimentConfig(
        HamiltonianTomography,
        [qubit],
        args={
            'schedule': sched,
            'widths': np.linspace(0., 2048., 17)
        }
    )

@register_post
def t_hx(runner, experiment_data):
    components = experiment_data.analysis_results('hamiltonian_components', block=False).value
    runner.program_data['target_rxtone_hamiltonian'] = components

@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_hcr_singlestate_template(runner, exp_type):
    """CR Hamiltonian tomography with a single control state and an offset Rx tone.

    Control state and cr_amp should be set through cr_rabi_init(state) and
    schedule.assign_parameters(), respectively, in args after the experiment configuration is
    instantiated.
    """
    from ..experiments.hamiltonian_tomography import HamiltonianTomography

    control2, target = runner.program_data['qubits'][1:]
    counter_amp = 2. / 2048 / runner.backend.dt / rabi_freq_per_amp(runner.backend, target)
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_frequency': runner.backend.qubit_properties(target).frequency,
        'cr_amp': Parameter('cr_amp'),
        'cr_sign_angle': 0.,
        'cr_stark_amp': 0.,
        'counter_amp': counter_amp,
        'counter_sign_angle': 0.,
        'counter_stark_amp': 0.
    }
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
        exp_type=exp_type
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_hcr_amplitude_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_hamiltonian import QutritCRHamiltonianTomographyScan

    control2, target = runner.program_data['qubits'][1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_frequency': runner.backend.qubit_properties(target).frequency,
        'cr_amp': Parameter('cr_amp'),
        'cr_sign_angle': 0.,
        'cr_stark_amp': 0.,
        'counter_amp': 0.,
        'counter_stark_amp': 0.
    }
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
            'poly_orders': poly_orders
        }
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_ucr_tomography_template(runner, exp_type):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    qubits = runner.program_data['qubits'][1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_frequency': Parameter('stark_frequency'),
        'cr_amp': Parameter('cr_amp'),
        'cr_sign_angle': 0.,
        'cr_stark_amp': Parameter('cr_stark_amp'),
        'cr_stark_sign_phase': Parameter('cr_stark_sign_phase'),
        'counter_amp': Parameter('counter_amp'),
        'counter_stark_amp': Parameter('counter_stark_amp')
    }
    schedule = runner.calibrations.get_schedule('cr', qubits, assign_params=assign_params)

    circuit = QuantumCircuit(2)
    circuit.append(Gate('cr', 2, []), [0, 1])
    circuit.add_calibration('cr', qubits, schedule)

    return ExperimentConfig(
        QutritQubitTomography,
        qubits,
        args={
            'circuit': circuit
        },
        exp_type=exp_type
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_sizzle_t_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRTargetStarkCal
    if (amp_pred := runner.program_data.get('sizzle_t_amp_pred')) is not None:
        amplitudes = np.linspace(0.5 * amp_pred, min(0.1, 1.5 * amp_pred), 10)
    else:
        amplitudes = None

    return ExperimentConfig(
        QutritCRTargetStarkCal,
        runner.program_data['qubits'][1:],
        args={'amplitudes': amplitudes}
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_sizzle_c2_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRControlStarkCal
    qubits = runner.program_data['qubits'][1:]
    if (amp_pred := runner.program_data.get('sizzle_c_amp_pred')) is not None:
        max_amp = 0.99 - runner.calibrations.get_parameter_value('cr_amp', qubits, 'cr')
        amplitudes = np.linspace(max(-max_amp, -1.5 * abs(amp_pred)),
                                 min(max_amp, 1.5 * abs(amp_pred)), 10)
    else:
        amplitudes = None

    return ExperimentConfig(
        QutritCRControlStarkCal,
        qubits,
        args={'amplitudes': amplitudes}
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_rcr_rotary(runner):
    from ..experiments.qutrit_qubit_cx.repeated_cr_rotary import RepeatedCRRotaryAmplitudeCal
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:]
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_crcr_rx_amp(runner):
    from ..experiments.qutrit_qubit_cx.rx_amp import CycledRepeatedCRRxAmplitudeCal
    return ExperimentConfig(
        CycledRepeatedCRRxAmplitudeCal,
        runner.program_data['qubits'][1:]
    )

@register_post
def c2t_crcr_rx_amp(runner, experiment_data):
    params = unp.nominal_values(experiment_data.analysis_results('unitary_line_fit_params',
                                                                 block=False).value)
    runner.program_data['crcr_angle_per_rx_amp'] = params[0] * np.sin(params[2]) * np.cos(params[3])

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_crcr_cr_width(runner):
    from ..experiments.qutrit_qubit_cx.cr_semifine_width import CycledRepeatedCRWidthCal
    qubits = runner.program_data['qubits'][1:]

    current_width = runner.calibrations.get_parameter_value('width', qubits, schedule='cr')
    if current_width != 0.:
        widths = np.linspace(current_width - 128, current_width + 128, 9)
        while widths[0] < 0.:
            widths += 16.
    else:
        widths = None

    return ExperimentConfig(
        CycledRepeatedCRWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'widths': widths,
        }
    )

@register_post
def c2t_crcr_cr_width(runner, experiment_data):
    params = unp.nominal_values(experiment_data.analysis_results('unitary_line_fit_params',
                                                                 block=False).value)
    runner.program_data['crcr_angle_per_dt'] = params[0] * np.sin(params[2]) * np.cos(params[3])

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_crcr_fine_rx_amp(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineRxAmpCal
    return ExperimentConfig(
        CycledRepeatedCRFineRxAmpCal,
        runner.program_data['qubits'][1:],
        args={
            'angle_per_amp': runner.program_data['crcr_angle_per_rx_amp']
        }
    )

@add_readout_mitigation(logical_qubits=[1], expval=True)
@register_exp
def c2t_crcr_fine_cr_width(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineCRWidthCal
    return ExperimentConfig(
        CycledRepeatedCRFineCRWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'angle_per_dt': runner.program_data['crcr_angle_per_dt']
        }
    )
