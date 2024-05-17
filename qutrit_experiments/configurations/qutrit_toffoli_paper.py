"""Experiments for taking data shown in the qutrit Toffoli paper."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit_experiments.calibration_management import ParameterValue
from qiskit_experiments.exceptions import CalibrationError

from qutrit_experiments.configurations.common import add_readout_mitigation, add_qpt_readout_mitigation
from qutrit_experiments.experiment_config import ExperimentConfig, register_exp
from qutrit_experiments.gates import P2Gate, QutritCCZGate, X12Gate, XplusGate, XminusGate
from qutrit_experiments.transpilation.qutrit_transpiler import (BASIS_GATES,
                                                                transpile_qutrit_circuits, make_instruction_durations)
from qutrit_experiments.transpilation.layout_and_translation import UndoLayout, generate_layout_passmanager
from qutrit_experiments.transpilation.dynamical_decoupling import DDCalculator
from qutrit_experiments.util.qutrit_toffoli import qutrit_toffoli_circuit

###########################
### Circuit definitions ###
###########################

def total_ccz_duration_nobc(runner):
    ecr_dur_c1c2 = runner.backend.target['ecr'][runner.qubits[:2]].calibration.duration
    ecr_dur_tc2 = runner.backend.target['ecr'][runner.qubits[2:0:-1]].calibration.duration
    x_dur = runner.backend.target['x'][runner.qubits[1:2]].calibration.duration

    alpha = x_dur + ecr_dur_tc2 + x_dur + ecr_dur_tc2
    duration = 2 * x_dur
    duration += ecr_dur_c1c2 + x_dur
    duration += alpha
    duration += 2 * x_dur + alpha
    duration += ecr_dur_c1c2
    duration += 2 * x_dur
    return duration

def total_ccz_duration_bc(runner):
    ecr_dur_c1c2 = runner.backend.target['ecr'][runner.qubits[:2]].calibration.duration
    ecr_dur_tc2 = runner.backend.target['ecr'][runner.qubits[2:0:-1]].calibration.duration
    x_dur = runner.backend.target['x'][runner.qubits[1:2]].calibration.duration

    alpha = x_dur + ecr_dur_tc2 + x_dur + ecr_dur_tc2
    duration = 2 * x_dur
    duration += ecr_dur_c1c2
    duration += alpha
    duration += 2 * x_dur + alpha
    duration += ecr_dur_c1c2
    t2 = duration
    duration += 2 * x_dur
    ref_delay = t2 - 3 * alpha - 2 * x_dur
    if ref_delay >= 0:
        duration += ref_delay + 2 * x_dur
    else:
        raise NotImplementedError('Basis-cycled CCZ needs negative delay')

    return duration

def xminusxplus_circuit(runner):
    x_dur = runner.backend.target['x'][runner.qubits[1:2]].calibration.duration

    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.append(X12Gate(), [0])
    circuit.delay(total_ccz_duration_nobc(runner) - 4 * x_dur, 0)
    circuit.append(X12Gate(), [0])
    circuit.x(0)

    return circuit

def xplus3_circuit(runner):
    x_dur = runner.backend.target['x'][runner.qubits[1:2]].calibration.duration
    delay = (total_ccz_duration_bc(runner) - 6 * x_dur) // 2

    circuit = QuantumCircuit(1)
    circuit.append(X12Gate(), [0])
    circuit.x(0)
    circuit.delay(delay, 0)
    circuit.append(X12Gate(), [0])
    circuit.x(0)
    circuit.delay(delay, 0)
    circuit.append(X12Gate(), [0])
    circuit.x(0)

    return circuit

def skeleton_cz_circuit_core(runner, ecr_tc2=False):
    """Skeleton CZ-like circuit."""
    instruction_durations = make_instruction_durations(runner.backend, runner.calibrations,
                                                       runner.qubits)
    def dur(gate, *qubits):
        return instruction_durations.get(gate, tuple(runner.qubits[iq] for iq in qubits))

    pulse_alignment = runner.backend.target.pulse_alignment
    ddapp = DDCalculator(runner.qubits, instruction_durations, pulse_alignment)

    x_dur = dur('x', 1)
    xplus_dur = dur('x12', 1) + x_dur
    ecr_dur_tc2 = dur('ecr', 2, 1)

    alpha = x_dur + ecr_dur_tc2 + x_dur + ecr_dur_tc2

    circuit = QuantumCircuit(3)

    # Reverse CX (missing the last X of c2 and a block of duration 2X on t)
    ddapp.append_dd(circuit, 0, 2 * alpha + xplus_dur, 5)
    circuit.append(X12Gate(), [1])
    if ecr_tc2:
        circuit.ecr(2, 1)
        circuit.delay(x_dur, 1)
        circuit.x(2)
        circuit.ecr(2, 1)
        circuit.x(2)
        circuit.append(P2Gate(-np.pi / 2.), [1])
        circuit.rz(-np.pi, 1)
    else:
        circuit.x(1)
        circuit.delay(alpha - xplus_dur, 1)
        ddapp.append_dd(circuit, 2, alpha, 2, distribution='right')

    circuit.append(X12Gate(), [1])
    circuit.x(1)
    circuit.delay(alpha - xplus_dur, 1)
    circuit.append(X12Gate(), [1])
    circuit.delay(x_dur, 2)
    ddapp.append_dd(circuit, 2, alpha - xplus_dur, distribution='left')

    return circuit

def skeleton_cz_circuit(runner, delta_cz=0., ecr_tc2=False):
    instruction_durations = make_instruction_durations(runner.backend, runner.calibrations,
                                                       runner.qubits)
    x_dur = instruction_durations.get('x', runner.qubits[2:])

    circuit = QuantumCircuit(3)

    if delta_cz:
        circuit.rz(-delta_cz, 1)

    circuit.compose(skeleton_cz_circuit_core(runner, ecr_tc2=ecr_tc2), inplace=True)
    circuit.x(1)
    circuit.delay(2 * x_dur, 2)

    return circuit

def skeleton_ccz_circuit(runner, delta_cz=0., delta_ccz=0., ecr_c1c2=False, ecr_tc2=False):
    """Skeleton CCZ-like circuit."""
    instruction_durations = make_instruction_durations(runner.backend, runner.calibrations,
                                                       runner.qubits)
    def dur(gate, *qubits):
        return instruction_durations.get(gate, tuple(runner.qubits[iq] for iq in qubits))

    pulse_alignment = runner.backend.target.pulse_alignment
    ddapp = DDCalculator(runner.qubits, instruction_durations, pulse_alignment)

    x_dur = dur('x', 1)
    x12_dur = dur('x12', 1)
    xplus_dur = dur('x12', 1) + x_dur
    ecr_dur_c1c2 = dur('ecr', 0, 1)
    cr_dur_c1c2 = (ecr_dur_c1c2 - x_dur) // 2
    ecr_dur_tc2 = dur('ecr', 2, 1)

    alpha = x_dur + ecr_dur_tc2 + x_dur + ecr_dur_tc2
    t2 = 2 * x_dur
    t2 += ecr_dur_c1c2
    t2 += alpha
    t2 += 2 * x_dur + alpha
    t2 += ecr_dur_c1c2
    refocusing_delay = t2 - 3 * alpha - 2 * x_dur
    if refocusing_delay < 0:
        raise NotImplementedError('Negative refocusing delay')

    circuit = QuantumCircuit(3)
    # X+X+ (last X in the next block)
    ddapp.append_dd(circuit, 0, 2 * xplus_dur + refocusing_delay, distribution='left')
    circuit.append(X12Gate(), [1])
    circuit.x(1)
    circuit.delay(refocusing_delay, 1)
    circuit.append(X12Gate(), [1])
    ddapp.append_dd(circuit, 2, 2 * xplus_dur + refocusing_delay, distribution='right')

    # ECR
    if ecr_c1c2:
        circuit.sx(1)
        circuit.append(P2Gate(-np.pi / 4.), [1])
        circuit.ecr(0, 1)
    else:
        circuit.x(1)
        circuit.delay(cr_dur_c1c2, 0)
        circuit.x(0)
        circuit.delay(cr_dur_c1c2, 0)
        circuit.delay(ecr_dur_c1c2, 1)
    ddapp.append_dd(circuit, 2, ecr_dur_c1c2 + x12_dur, 2)

    if delta_cz:
        circuit.rz(-delta_cz, 1)

    # Reverse CX (last X in the next block)
    circuit.compose(skeleton_cz_circuit_core(runner, ecr_tc2=ecr_tc2), inplace=True)
    circuit.x(2)
    circuit.delay(x_dur, 2)

    # ECR
    if ecr_c1c2:
        circuit.rz(np.pi, 1)
        circuit.sx(1)
        circuit.rz(-np.pi, 1)
        circuit.append(P2Gate(3. * np.pi / 4.), [1])
        circuit.ecr(0, 1)
    else:
        circuit.x(1)
        circuit.delay(cr_dur_c1c2, 0)
        circuit.x(0)
        circuit.delay(cr_dur_c1c2, 0)
        circuit.delay(ecr_dur_c1c2, 1)

    circuit.delay(xplus_dur, 0)
    circuit.append(X12Gate(), [1])
    circuit.x(1)
    ddapp.append_dd(circuit, 2, ecr_dur_c1c2 + x_dur, 2)
    circuit.x(2)

    if delta_ccz:
        circuit.rz(-delta_ccz, 1)

    return circuit

def translate(circuit, runner):
    basis_gates = runner.backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    pms = {
        'layout': generate_layout_passmanager(runner.qubits, runner.backend.coupling_map),
        'translation': PassManager([BasisTranslator(sel, basis_gates)]),
        'layin': PassManager([UndoLayout(runner.qubits)])
    }
    return StagedPassManager(list(pms.keys()), **pms).run(circuit)

def id0_circuit(runner):
    """Circuit with only the Xpluses and DDs."""
    delta_cz = runner.calibrations.get_parameter_value('delta_cz_id', runner.qubits)
    delta_ccz = runner.calibrations.get_parameter_value('delta_ccz_id0', runner.qubits)
    circuit = skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz)
    return translate(circuit, runner)

def id1_circuit(runner):
    """Circuit with Xpluses, DDs, and ECR(c1,c2)."""
    delta_cz = runner.calibrations.get_parameter_value('delta_cz_id', runner.qubits)
    delta_ccz = runner.calibrations.get_parameter_value('delta_ccz_id1', runner.qubits)
    circuit = skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz, ecr_c1c2=True)
    return translate(circuit, runner)

def cz_circuit(runner):
    """Circuit with Xpluses, DDs, and ECR(c1,c2)."""
    delta_cz = runner.calibrations.get_parameter_value('delta_cz', runner.qubits)
    delta_ccz = runner.calibrations.get_parameter_value('delta_ccz_cz', runner.qubits)
    circuit = skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz, ecr_tc2=True)
    return translate(circuit, runner)

#######################################
### Calibrations for error analysis ###
#######################################

@register_exp
@add_readout_mitigation
def cz_id_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    param_name = 'delta_cz_id'
    try:
        delta_cz = runner.calibrations.get_parameter_value(param_name, runner.qubits)
    except CalibrationError:
        runner.calibrations.add_parameter_value(ParameterValue(0.), param_name, runner.qubits)
        delta_cz = 0.
    circuit = translate(skeleton_cz_circuit(runner, delta_cz=delta_cz), runner)

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': param_name
        },
        analysis_options={'outcome': '010'}
    )

@register_exp
@add_readout_mitigation
def ccz_id0_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    param_name = 'delta_ccz_id0'
    delta_cz = runner.calibrations.get_parameter_value('delta_cz_id', runner.qubits)
    try:
        delta_ccz = runner.calibrations.get_parameter_value(param_name, runner.qubits)
    except CalibrationError:
        delta_ccz = runner.calibrations.add_parameter_value(ParameterValue(0.), param_name,
                                                            runner.qubits)

    circuit = translate(skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz),
                        runner)

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': param_name
        },
        analysis_options={'outcome': '010'}
    )

@register_exp
@add_readout_mitigation
def ccz_id1_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    param_name = 'delta_ccz_id1'
    delta_cz = runner.calibrations.get_parameter_value('delta_cz_id', runner.qubits)
    try:
        delta_ccz = runner.calibrations.get_parameter_value(param_name, runner.qubits)
    except CalibrationError:
        delta_ccz = runner.calibrations.add_parameter_value(ParameterValue(0.), param_name,
                                                            runner.qubits)

    circuit = translate(skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz,
                                             ecr_c1c2=True),
                        runner)

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': param_name
        },
        analysis_options={'outcome': '010'}
    )

@register_exp
@add_readout_mitigation
def ccz_cz_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    param_name = 'delta_ccz_cz'
    delta_cz = runner.calibrations.get_parameter_value('delta_cz', runner.qubits)
    try:
        delta_ccz = runner.calibrations.get_parameter_value(param_name, runner.qubits)
    except CalibrationError:
        delta_ccz = runner.calibrations.add_parameter_value(ParameterValue(0.), param_name,
                                                            runner.qubits)

    circuit = translate(skeleton_ccz_circuit(runner, delta_cz=delta_cz, delta_ccz=delta_ccz,
                                             ecr_tc2=True),
                        runner)

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': param_name
        },
        analysis_options={'outcome': '010'}
    )


#################################
### Basis-cycle demonstrators ###
#################################

@register_exp
@add_readout_mitigation
def c2phase_xminusxplus(runner):
    from qutrit_experiments.experiments.phase_table import DiagonalCircuitPhaseShift

    return ExperimentConfig(
        DiagonalCircuitPhaseShift,
        runner.qubits[1:2],
        args={
            'circuit': xminusxplus_circuit(runner),
            'state': [None],
        }
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_xminusxplus(runner):
    from qutrit_experiments.experiments.process_tomography import CircuitTomography

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits[1:2],
        args={
            'circuit': xminusxplus_circuit(runner),
            'target_circuit': QuantumCircuit(1)
        },
        run_options={'shots': 2000},
        analysis_options={'target_bootstrap_samples': 100}
    )

@register_exp
@add_readout_mitigation
def c2phase_xplus3(runner):
    from qutrit_experiments.experiments.phase_table import DiagonalCircuitPhaseShift

    return ExperimentConfig(
        DiagonalCircuitPhaseShift,
        runner.qubits[1:2],
        args={
            'circuit': xplus3_circuit(runner),
            'state': [None],
        }
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_xplus3(runner):
    from qutrit_experiments.experiments.process_tomography import CircuitTomography

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits[1:2],
        args={
            'circuit': xplus3_circuit(runner),
            'target_circuit': QuantumCircuit(1)
        },
        run_options={'shots': 2000},
        analysis_options={'target_bootstrap_samples': 100}
    )

######################
### Error analysis ###
######################

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_id0(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': id0_circuit(runner)
        }
    )

@register_exp
@add_readout_mitigation
def phasetable_id0(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': id0_circuit(runner)
        }
    )

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_id1(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': id1_circuit(runner)
        }
    )

@register_exp
@add_readout_mitigation
def phasetable_id1(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': id1_circuit(runner)
        }
    )

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_cz(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': cz_circuit(runner)
        }
    )

@register_exp
@add_readout_mitigation
def phasetable_cz(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': cz_circuit(runner)
        }
    )
