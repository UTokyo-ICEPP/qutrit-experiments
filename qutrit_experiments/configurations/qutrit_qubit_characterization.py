# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
# pylint: disable=unexpected-keyword-arg, redundant-keyword-arg
"""Experiment configurations for Toffoli gate calibration."""

import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit_experiments.framework import ExperimentData
from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..runners import ExperimentsRunner
from ..util.pulse_area import rabi_cycles_per_area
from .common import add_readout_mitigation


@add_readout_mitigation(logical_qubits=[1])
def sizzle_template(runner: ExperimentsRunner, exp_type: str) -> ExperimentConfig:
    """Template for SiZZle measurement at a single frequency and amplitude combination.

    Frequency, amplitudes should be set in args.
    """
    from ..experiments.qutrit_qubit.sizzle import SiZZle
    qubits = runner.qubits[1:]

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


@register_exp
@add_readout_mitigation(logical_qubits=[1])
def zzramsey(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.zzramsey import QutritZZRamsey
    return ExperimentConfig(
        QutritZZRamsey,
        runner.qubits[1:],
        args={
            'delays': np.linspace(0., 4.e-7, 6),
            'osc_freq': 2.e+6
        },
        run_options={'shots': 2000}
    )


@register_post
def zzramsey(runner: ExperimentsRunner, experiment_data: ExperimentData) -> None:
    omega_zs = experiment_data.analysis_results('omega_zs', block=False).value
    runner.program_data['static_omega_zs'] = omega_zs


@add_readout_mitigation(logical_qubits=[1])
@register_exp
def sizzle_frequency_scan(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.sizzle import SiZZleFrequencyScan

    control2, target = runner.qubits[1:]
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
def hcr_template(runner: ExperimentsRunner, exp_type: str) -> ExperimentConfig:
    """CR HT with undefined amplitude."""
    from ..experiments.qutrit_qubit.qutrit_cr_hamiltonian import QutritCRHamiltonianTomography

    control2, target = runner.qubits[1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_freq': runner.backend.qubit_properties(target).frequency * runner.backend.dt,
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


@register_exp
@add_readout_mitigation(expval=True)
def t_hx(runner: ExperimentsRunner) -> ExperimentConfig:
    """Target qubit Rx tone Hamiltonian with amplitude set for two cycles in 2048 dt."""
    from ..experiments.hamiltonian_tomography import HamiltonianTomography

    qubit = runner.qubits[2]
    width = Parameter('width')
    sigma = 64
    rsr = 2
    duration = width + sigma * rsr * 2
    cycles_per_amp = 2048. * rabi_cycles_per_area(runner.backend, qubit)
    amp = 2. / cycles_per_amp
    angle = runner.calibrations.get_parameter_value('counter_base_angle',
                                                    runner.qubits[1:],
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
def t_hx(runner: ExperimentsRunner, experiment_data: ExperimentData) -> None:
    components = experiment_data.analysis_results('hamiltonian_components', block=False).value
    runner.program_data['target_rxtone_hamiltonian'] = components


@add_readout_mitigation(logical_qubits=[1], expval=True)
def hcr_singlestate_template(runner: ExperimentsRunner, exp_type: str) -> ExperimentConfig:
    """CR Hamiltonian tomography with a single control state and an offset Rx tone.

    Control state and cr_amp should be set through cr_rabi_init(state) and
    schedule.assign_parameters(), respectively, in args after the experiment configuration is
    instantiated.
    """
    from ..experiments.hamiltonian_tomography import HamiltonianTomography
    from ..experiments.gs_rabi import GSRabi

    control2, target = runner.qubits[1:]
    cycles_per_amp = 2048. * rabi_cycles_per_area(runner.backend, target)
    counter_amp = 2. / cycles_per_amp
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_freq': runner.backend.qubit_properties(target).frequency * runner.backend.dt,
        'cr_amp': Parameter('cr_amp'),
        'cr_sign_angle': 0.,
        'cr_stark_amp': 0.,
        'counter_amp': counter_amp,
        'counter_sign_angle': 0.,
        'counter_stark_amp': 0.
    }
    schedule = runner.calibrations.get_schedule('cr', qubits=[control2, target],
                                                assign_params=assign_params)

    def rabi_init(*args, **kwargs):
        exp = GSRabi(*args, **kwargs)
        exp.set_experiment_options(measured_logical_qubit=1)
        return exp

    return ExperimentConfig(
        HamiltonianTomography,
        [control2, target],
        args={
            'schedule': schedule,
            'rabi_init': rabi_init
        },
        exp_type=exp_type
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def hcr_amplitude_scan(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.qutrit_cr_hamiltonian import QutritCRHamiltonianTomographyScan

    control2, target = runner.qubits[1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_freq': runner.backend.qubit_properties(target).frequency * runner.backend.dt,
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
def ucr_tomography_template(runner: ExperimentsRunner, exp_type: str) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    qubits = runner.qubits[1:]
    assign_params = {
        'width': Parameter('width'),
        'margin': 0.,
        'stark_freq': Parameter('stark_freq'),
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
