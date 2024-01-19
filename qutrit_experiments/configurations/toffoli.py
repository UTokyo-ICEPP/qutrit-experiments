# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.data_processing import DataProcessor, Probability

from ..data_processing import ReadoutMitigation
from ..experiment_config import BatchExperimentConfig, ExperimentConfig, register_exp, register_post
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

def add_readout_mitigation(gen):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    @wraps(gen)
    def converted_gen(runner):
        config = gen(runner)
        if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
            logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation for %s',
                           gen.__name__)
            return config
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
            probability_pos = next(i for i, node in enumerate(processor._nodes)
                                   if isinstance(node, Probability))
            processor._nodes.insert(probability_pos, ReadoutMitigation(matrix))
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
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator').value
    prog_data = runner.program_data.setdefault('readout_assignment_matrices', {})
    for combination in [qubits[0:1], qubits[1:2], qubits[2:3], qubits[:2], qubits[1:3], qubits]:
        prog_data[combination] = mitigator.assignment_matrix(combination)

@register_exp
@add_readout_mitigation
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
    for freq in np.linspace(min(resonances.values()) - 1.e+8, max(resonances.values()) + 1.e+8, 20):
        if all(abs(freq - res) > 1.e+7 for res in resonances.values()):
            frequencies.append(freq)

    cr_angle = runner.calibrations.get_parameter_value('cr_angle', [control2, target],
                                                       schedule='cr')
    return ExperimentConfig(
        SiZZleFrequencyScan,
        [control2, target],
        args={
            'frequencies': frequencies,
            'delays': np.linspace(0., 4.e-7, 16),
            'osc_freq': 5.e+6,
            'control_phase_offset': cr_angle
        },
        experiment_options={'frequencies_of_interest': resonances}
    )

@register_post
def c2t_sizzle_frequency_scan(runner, data):
    frequencies = np.empty(len(data.child_data()) - 1, dtype=float)
    shifts = np.empty(frequencies.shape + (3,), dtype=float)
    component_index = data.metadata["component_child_index"]
    for ichild, child_index in enumerate(component_index[:-1]):
        child_data = data.child_data(child_index)
        frequencies[ichild] = child_data.metadata['frequency']
        shifts[ichild] = unp.nominal_values(child_data.analysis_results('omega_zs').value)

    runner.program_data['sizzle_frequencies'] = frequencies
    runner.program_data['sizzle_shifts'] = shifts

@register_exp
@add_readout_mitigation
def c2t_hcr(runner):
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
        }
    )


@register_exp
@add_readout_mitigation
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
        analysis_options={'poly_orders': poly_orders}
    )
