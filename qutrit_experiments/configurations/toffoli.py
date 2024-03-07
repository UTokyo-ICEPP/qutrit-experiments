# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter


from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..experiments.qutrit_qubit_cx.util import (RCRType, get_cr_schedules, make_cr_circuit,
                                                make_crcr_circuit)
from ..gates import QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES, RZ12Gate, X12Gate
from ..transpilation.layout_and_translation import generate_translation_passmanager
from ..transpilation.qutrit_transpiler import make_instruction_durations
from ..transpilation.rz import ConsolidateRZAngle
from ..util.pulse_area import rabi_freq_per_amp
from .common import add_readout_mitigation, qubits_assignment_error, qubits_assignment_error_post
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
    qutrit_sx12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
)

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


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
    qutrit_sx12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
]
for func in qutrit_functions:
    register_single_qutrit_exp(func)


@register_exp
@wraps(qubits_assignment_error)
def qubits_assignment_error_func(runner):
    return qubits_assignment_error(runner, runner.program_data['qubits'])

register_post(qubits_assignment_error_post, exp_type='qubits_assignment_error')

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_cr_unitaries(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    qubits = runner.program_data['qubits'][1:]
    cr_circuit = make_cr_circuit(qubits, runner.calibrations)

    return ExperimentConfig(
        QutritQubitTomography,
        qubits,
        args={'circuit': cr_circuit}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_unitaries(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    qubits = tuple(runner.program_data['qubits'][1:])
    crcr_circuit = make_crcr_circuit(qubits, runner.calibrations)

    return ExperimentConfig(
        QutritQubitTomography,
        qubits,
        args={'circuit': crcr_circuit}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_cr_rough_width(runner):
    """Few-sample CR UT to measure ωx to find a rough estimate for the CR width in CRCR."""
    from ..experiments.qutrit_qubit_cx.cr_width import CRRoughWidthCal
    return ExperimentConfig(
        CRRoughWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'widths': np.arange(128., 384., 64.)
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_sizzle_t_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRTargetStarkCal
    return ExperimentConfig(
        QutritCRTargetStarkCal,
        runner.program_data['qubits'][1:]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_sizzle_c2_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRControlStarkCal
    return ExperimentConfig(
        QutritCRControlStarkCal,
        runner.program_data['qubits'][1:]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_cr_width(runner):
    from ..experiments.qutrit_qubit_cx.cr_width import CycledRepeatedCRWidthCal
    qubits = runner.program_data['qubits'][1:]

    current_width = runner.calibrations.get_parameter_value('width', qubits, schedule='cr')
    if current_width != 0.:
        widths = np.linspace(current_width - 128, current_width + 128, 5)
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

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_rotary(runner):
    from ..experiments.qutrit_qubit_cx.rotary import (CycledRepeatedCRRotaryAmplitudeCal,
                                                      rotary_angle_per_amp)
    qubits = runner.program_data['qubits'][1:]
    angle_per_amp = rotary_angle_per_amp(runner.backend, runner.calibrations, qubits)
    # crcr_rotary_test_angles are the rotary angles with the current CR parameters
    if (angles := runner.program_data.get('crcr_rotary_test_angles')) is None:
        # Scan rotary amplitudes expected to generate +-1 rad rotations within one CR pulse
        angles = np.linspace(-1., 1., 8)

    return ExperimentConfig(
        CycledRepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:],
        args={'amplitudes': angles / angle_per_amp}
    )

@register_post
def c2t_crcr_rotary(runner, experiment_data):
    """Identify the target Rx angle for offset_rx."""
    # Which test amplitude was selected?
    rotary_amp = experiment_data.analysis_results('rotary_amp', block=False).value
    selected_amp_idx = int(np.argmin(
        np.abs(experiment_data.metadata['scan_values'][0] - rotary_amp)
    ))
    unitaries = unp.nominal_values(
        experiment_data.analysis_results('unitary_parameters', block=False).value[selected_amp_idx]
    )
    runner.program_data['crcr_unitaries_rough'] = unitaries

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_angle_width_rate(runner):
    """Measure the X rotation angles in 0 and 1 with the rotary."""
    from ..experiments.qutrit_qubit_cx.cr_width import CycledRepeatedCRWidth
    qubits = runner.program_data['qubits'][1:]

    cr_schedules = get_cr_schedules(runner.calibrations, qubits,
                                    free_parameters=['width', 'margin'])
    rcr_type = RCRType(runner.calibrations.get_parameter_value('rcr_type', qubits))

    # Frequency (cycles / clock) from the rotary tone (should dominate)
    rotary_amp = runner.calibrations.get_parameter_value('counter_amp', qubits, 'cr')
    rotary_freq = rabi_freq_per_amp(runner.backend, qubits[1]) * runner.backend.dt * rotary_amp
    # CRCR frequency is roughly (rotary+rotary)*(1+1-1) = 2*rotary
    # Aim for the width scan range of +-0.4 cycles
    current_width = runner.calibrations.get_parameter_value('width', qubits, schedule='cr')
    widths = np.linspace(current_width - 0.2 / rotary_freq, current_width + 0.2 / rotary_freq, 5)
    if widths[0] < 0.:
        widths += -widths[0]

    return ExperimentConfig(
        CycledRepeatedCRWidth,
        runner.program_data['qubits'][1:],
        args={
            'cr_schedules': cr_schedules,
            'rcr_type': rcr_type,
            'widths': widths,
        }
    )

@register_post
def c2t_crcr_angle_width_rate(runner, experiment_data):
    # d|θ_1x - θ_0x|/dt
    fit_params = experiment_data.analysis_results('unitary_linear_fit_params', block=False).value
    slope, _, psi, phi = np.stack([unp.nominal_values(fit_params[ic]) for ic in range(2)], axis=1)
    runner.program_data['crcr_angle_per_width'] = slope * np.sin(psi) * np.cos(phi)

@register_exp
@add_readout_mitigation
def c2t_crcr_rx_amp(runner):
    from ..experiments.qutrit_qubit_cx.rx_amp import SimpleRxAmplitudeCal
    qubits = (runner.program_data['qubits'][2],)
    x_sched = runner.backend.defaults().instruction_schedule_map.get('x', qubits)
    pi_amp = x_sched.instructions[0][1].pulse.amp

    return ExperimentConfig(
        SimpleRxAmplitudeCal,
        qubits,
        args={
            'target_angle': -runner.program_data['crcr_unitaries_rough'][0, 0], # X of c=0
            'amplitudes': np.linspace(-pi_amp, pi_amp, 32)
        }
    )

@register_post
def c2t_crcr_rx_amp(runner, experiment_data):
    angular_rate = experiment_data.analysis_results('rabi_rate', block=False).value.n * twopi
    runner.program_data['crcr_angle_per_rx_amp'] = angular_rate

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_crcr_fine(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineCal
    return ExperimentConfig(
        CycledRepeatedCRFineCal,
        runner.program_data['qubits'][1:],
        args={
            'width_rate': runner.program_data['crcr_angle_per_width'],
            'amp_rate': runner.program_data['crcr_angle_per_rx_amp']
        }
    )

@register_exp
@add_readout_mitigation
def toffoli_qpt_default(runner):
    from ..experiments.process_tomography import CircuitTomography

    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.program_data['qubits'],
        args={'circuit': circuit, 'target_circuit': circuit},
        experiment_options={'need_translation': True}
    )

@register_exp
@add_readout_mitigation
def toffoli_qpt_bare(runner):
    from ..experiments.process_tomography import CircuitTomography

    # Assuming ECR acts on c1->c2 (no layout applied; translator assumes q0 is the control)
    cx_circuit = QuantumCircuit(2)
    cx_circuit.cx(0, 1)
    pm = generate_translation_passmanager(runner.backend.operation_names)
    pm.append(ConsolidateRZAngle())
    cx_circuit = pm.run(cx_circuit)

    c2t = tuple(runner.program_data['qubits'][1:])
    cx_sign = runner.calibrations.get_parameter_value('qutrit_qubit_cx_sign', c2t)

    circuit = QuantumCircuit(3)
    circuit.x(1)
    circuit.append(X12Gate(), [1])
    circuit.compose(cx_circuit, [0, 1], inplace=True)
    circuit.compose(make_crcr_circuit(c2t, runner.calibrations), [1, 2], inplace=True)
    circuit.rz(cx_sign * np.pi / 3., 1)
    circuit.append(RZ12Gate(-cx_sign * np.pi / 3.), [1])
    circuit.compose(cx_circuit, [0, 1], inplace=True)
    circuit.append(X12Gate(), [1])
    circuit.x(1)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.program_data['qubits'],
        args={'circuit': circuit, 'target_circuit': target_circuit}
    )

@register_exp
@add_readout_mitigation
def toffoli_qpt_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    qubits = tuple(runner.program_data['qubits'])

    # Assuming ECR acts on c1->c2 (no layout applied; translator assumes q0 is the control)
    cx_circuit = QuantumCircuit(2)
    cx_circuit.cx(0, 1)
    pm = generate_translation_passmanager(runner.backend.operation_names)
    pm.append(ConsolidateRZAngle())
    cx_circuit = pm.run(cx_circuit)

    c2t = qubits[1:]
    cx_sign = runner.calibrations.get_parameter_value('qutrit_qubit_cx_sign', c2t)

    u_qutrit = QuantumCircuit(3)
    u_qutrit.compose(cx_circuit, [0, 1], inplace=True)
    u_qutrit.compose(make_crcr_circuit(c2t, runner.calibrations), [1, 2], inplace=True)
    u_qutrit.rz(cx_sign * np.pi / 3., 1)
    u_qutrit.append(RZ12Gate(-cx_sign * np.pi / 3.), [1])
    u_qutrit.compose(cx_circuit, [0, 1], inplace=True)

    instruction_durations = make_instruction_durations(runner.backend, runner.calibrations,
                                                       qubits=qubits)
    basis_gates = runner.backend.basis_gates + [
        inst.gate_name for inst in QUTRIT_PULSE_GATES + QUTRIT_VIRTUAL_GATES
    ]
    u_qutrit = transpile(u_qutrit, runner.backend, initial_layout=qubits, optimization_level=0,
                         basis_gates=basis_gates, scheduling_method='alap',
                         instruction_durations=instruction_durations)

    # Define a placeholder gate and implement delay-setting as a transpilation pass


    circuit = QuantumCircuit(3)
    circuit.append(X12Gate(), [1])
    circuit.x(1)
    circuit.delay(Parameter('qutrit_refocusing_delay'))
    circuit.append(X12Gate(), [1])
    circuit.x(1)

    circuit.append(X12Gate(), [1])
    circuit.x(1)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.program_data['qubits'],
        args={'circuit': circuit, 'target_circuit': target_circuit}
    )
