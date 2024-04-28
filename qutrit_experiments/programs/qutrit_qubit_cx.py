import logging
from typing import Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit_experiments.data_processing import (BasisExpectationValue, DataProcessor,
                                                MarginalizeCounts, Probability)

from ..experiment_config import experiments
from ..runners import ExperimentsRunner
from ..util.sizzle import sizzle_hamiltonian_shifts

from ..experiment_config import ExperimentConfig
from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography
from ..calibrations import get_qutrit_pulse_gate
from ..configurations.common import configure_readout_mitigation
from ..gates import QutritQubitCXType

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner,
    cx_type: int = QutritQubitCXType.CRCR,
    refresh_readout_error: bool = True,
    qutrit_qubit_index: Optional[tuple[int, int]] = None
):
    if qutrit_qubit_index is not None:
        runner_qubits = tuple(runner.qubits)
        runner.qubits = [runner.qubits[idx] for idx in qutrit_qubit_index]

    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    if cx_type == QutritQubitCXType.REVERSE:
        calibrations.add_parameter_value(int(cx_type), 'rcr_type', runner.qubits)
        runner.run_experiment('tc2_cr_rotary_delta')
        if qutrit_qubit_index is not None:
            runner.qubits = runner_qubits
        return

    # Find the amplitude that does not disrupt the |2> state too much
    runner.run_experiment('cr_initial_amp')

    # Determine the RCR type to use and corresponding CR width and Rx offset angle
    rough_width_data = runner.run_experiment('cr_rough_width')

    # Eliminate the y component of RCR non-participating state
    runner.run_experiment('cr_angle')

    eliminate_z(runner, rough_width_data)

    # Calculate the amplitude for CX given the width
    runner.run_experiment('rcr_rough_cr_amp')

    # Minimize the y and z components of RCR
    runner.run_experiment('rcr_rotary_amp')

    # Fine calibration
    runner.run_experiment('crcr_fine_scanbased')

    # CR unitaries
    cr_data = run_unitaries(runner, 'cr_unitaries')
    prep_unitaries = cr_data.analysis_results('prep_parameters').value

    # RCR unitaries
    run_unitaries(runner, 'rcr_unitaries', prep_unitaries)

    backend = runner.backend
    calibrations = runner.calibrations

    rcr_type = calibrations.get_parameter_value('rcr_type', runner.qubits)
    x12 = get_qutrit_pulse_gate('x12', runner.qubits[0], calibrations, target=backend.target)
    x_c = calibrations.get_schedule('x', runner.qubits[0])
    x_t = calibrations.get_schedule('x', runner.qubits[1])
    def x_dd():
        with pulse.align_left():
            pulse.call(x_c)
            pulse.call(x_t)
            pulse.call(x_t)

    def x12_dd():
        with pulse.align_left():
            pulse.call(x12)
            pulse.call(x_t)
            pulse.call(x_t)

    control_channel = backend.control_channel(runner.qubits)[0]
    target_drive_channel = backend.drive_channel(runner.qubits[1])
    cr = calibrations.get_schedule('cr', runner.qubits)

    # RCR in CRCR
    with pulse.build(name='rcr', default_alignment='sequential') as sched:
        if rcr_type == 2:
            x12_dd()
            pulse.call(cr)
            x_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.call(cr)
        else:
            pulse.call(cr)
            x12_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.call(cr)
            x_dd()

    circuit = QuantumCircuit(2)
    circuit.append(Gate('rcr', 2, []), [0, 1])
    circuit.add_calibration('rcr', runner.qubits, sched)
    run_unitaries(runner, 'rcrplus_unitaries', prep_unitaries, circuit)

    with pulse.build(name='rcr', default_alignment='sequential') as sched:
        with pulse.phase_offset(np.pi, control_channel):
            if rcr_type == 2:
                x12_dd()
                with pulse.phase_offset(np.pi, target_drive_channel):
                    pulse.call(cr)
                x_dd()
                pulse.call(cr)
            else:
                with pulse.phase_offset(np.pi, target_drive_channel):
                    pulse.call(cr)
                x12_dd()
                pulse.call(cr)
                x_dd()

    circuit = QuantumCircuit(2)
    circuit.append(Gate('rcr', 2, []), [0, 1])
    circuit.add_calibration('rcr', runner.qubits, sched)
    run_unitaries(runner, 'rcrminus_unitaries', prep_unitaries, circuit)

    # Validate
    for exp_type in ['crcr_unitaries', 'cx_unitaries']:
        #runner.run_experiment(exp_type)
        run_unitaries(runner, exp_type, prep_unitaries)

    if qutrit_qubit_index is not None:
        runner.qubits = runner_qubits


def eliminate_z(runner, rough_width_data):
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    fit_params = rough_width_data.analysis_results('simul_fit_params').value[rcr_type]
    omega_z = fit_params[0] * unp.cos(fit_params[2]) / runner.backend.dt
    if abs(omega_z.n) > omega_z.std_dev:
        # Set the target Stark frequency from observed sign of Z
        frequency, amp = get_stark_params(runner.backend, runner.qubits, rcr_type, omega_z.n)
        runner.calibrations.add_parameter_value(frequency * runner.backend.dt, 'stark_freq',
                                                runner.qubits, 'cr')
        # Eliminate the z component of RCR non-participating state
        config = experiments['cr_counter_stark_amp'](runner)
        config.args['amplitudes'] = np.linspace(0., amp * 2., 8)
        runner.run_experiment(config)


def get_stark_params(
    backend: Backend,
    qubits: tuple[int, int],
    control_state: int,
    omega_z: float
) -> float:
    frequencies = nonresonant_frequencies(backend, qubits[1])
    hvars = backend.configuration().hamiltonian['vars']
    amp = 0.08
    shifts_quditbasis = sizzle_hamiltonian_shifts(hvars, qubits, (0., amp), frequencies)[..., 1]
    # shifts is in [Izζ][Iz] basis
    qudit_to_control = np.array([[1., 1., 0.], [1., -1., 1.], [1., 0., -1.]])[control_state]
    shifts = np.einsum('i,fi->f', qudit_to_control, shifts_quditbasis)

    # Align the sign of shifts so we always deal with maxima
    shifts *= -np.sign(omega_z)
    targ = abs(omega_z)
    if np.amax(shifts) < targ:
        # All shifts are smaller than omega_z - increase the amplitude
        return frequencies[np.argmax(shifts)], amp * np.sqrt(targ / np.amax(shifts))
    if not np.any(np.isclose(shifts, targ, atol=5.e+4)):
        # Shifts are either too large or too small - use the minimum above omega_z
        shifts = np.where(shifts > targ, shifts, np.inf)
        return frequencies[np.argmin(shifts)], amp * np.sqrt(targ / np.amin(shifts))
    # Else, return the frequency & amp closest to omega_z
    idx = np.argmin(np.abs(shifts - targ))
    return frequencies[idx], amp


def nonresonant_frequencies(
    backend: Backend,
    qubit: int,
    distance: int = 1,
    npoints: int = 400
) -> np.ndarray:
    """Return disjoint frequency intervals avoiding +-20 MHz resonances of neighboring qubits."""
    neighbors = np.nonzero(backend.coupling_map.distance_matrix[qubit] <= distance)[0]
    qubit_prop = backend.qubit_properties(qubit)
    qubit_props = [backend.qubit_properties(int(q)) for q in neighbors]
    qubit_frequencies = np.array([p.frequency for p in qubit_props])
    anharmonicities = np.array([p.anharmonicity for p in qubit_props])
    resonances = np.sort(np.concatenate([qubit_frequencies, qubit_frequencies + anharmonicities]))
    test_points = np.linspace(qubit_prop.frequency + qubit_prop.anharmonicity + 20.e+6,
                              qubit_prop.frequency + 100.e+6,
                              npoints)
    mask = np.any((test_points[:, None] > resonances[None, :] - 20.e+6)
                  & (test_points[:, None] < resonances[None, :] + 20.e+6), axis=1)
    return test_points[np.logical_not(mask)]


def run_unitaries(runner, config, prep_unitaries=None, circuit=None, block=False):
    if isinstance(config, str):
        if circuit is None:
            config = experiments[config](runner)
        else:
            dp_nodes = [MarginalizeCounts({0}), Probability('1'), BasisExpectationValue()]
            config = ExperimentConfig(
                QutritQubitTomography,
                runner.qubits,
                args={'circuit': circuit, 'measure_qutrit': True},
                run_options={'shots': 8000, 'rep_delay': 5.e-4},
                analysis_options={
                    'data_processor': DataProcessor('counts', dp_nodes)
                },
                exp_type=config
            )
            configure_readout_mitigation(runner, config, logical_qubits=[1], expval=True)
    config.analysis_options['qutrit_assignment_matrix'] = runner.program_data['qutrit_assignment_matrix'][runner.qubits[0]]
    if prep_unitaries:
        config.args['measure_preparations'] = False
        config.analysis_options['prep_unitaries'] = prep_unitaries

    return runner.run_experiment(config, block_for_results=block or (prep_unitaries is None))
