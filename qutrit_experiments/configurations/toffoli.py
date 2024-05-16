# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
import numpy as np
from qiskit import QuantumCircuit, schedule
from qiskit_experiments.calibration_management import ParameterValue
from qiskit_experiments.exceptions import CalibrationError


from ..experiment_config import ExperimentConfig, register_exp
from ..gates import (QutritCCZGate, QutritQubitCXType, QutritQubitCXGate, QutritQubitCZGate,
                     X12Gate, XminusGate, XplusGate)
from ..transpilation.layout_and_translation import generate_translation_passmanager
from ..transpilation.qutrit_toffoli import reverse2q_3q_decomposition_circuit
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.rz import ConsolidateRZAngle
from ..util.qutrit_toffoli import qutrit_toffoli_circuit, qutrit_toffoli_translator
from ..util.transforms import circuit_to_pulse_circuit
from .common import add_readout_mitigation, add_qpt_readout_mitigation


@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c1c2_cr_rotary_delta(runner):
    from ..experiments.qutrit_qubit.rotary_stark_shift import RotaryStarkShiftPhaseCal
    return ExperimentConfig(
        RotaryStarkShiftPhaseCal,
        runner.qubits[:2],
        analysis_options={'outcome': '1'}
    )

@register_exp
@add_readout_mitigation
def cz_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    try:
        delta_cz = runner.calibrations.get_parameter_value('delta_cz', runner.qubits)
    except CalibrationError:
        runner.calibrations.add_parameter_value(ParameterValue(0.), 'delta_cz', runner.qubits)
        delta_cz = 0.

    inst_durations = make_instruction_durations(runner.backend, runner.calibrations, runner.qubits)
    circuit = reverse2q_3q_decomposition_circuit('qutrit_qubit_cz', runner.qubits, inst_durations,
                                                 delta_cz=delta_cz)
    pm = qutrit_toffoli_translator(runner.backend, runner.calibrations, runner.qubits)
    circuit = pm.run(circuit)

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': 'delta_cz'
        },
        analysis_options={'outcome': '010'}
    )

@register_exp
@add_readout_mitigation
def ccz_c2_phase(runner):
    from ..experiments.phase_table import DiagonalPhaseCal

    try:
        runner.calibrations.get_parameter_value('delta_ccz', runner.qubits)
    except CalibrationError:
        runner.calibrations.add_parameter_value(ParameterValue(0.), 'delta_ccz', runner.qubits)

    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())

    return ExperimentConfig(
        DiagonalPhaseCal,
        runner.qubits,
        args={
            'circuit': circuit,
            'state': [0, None, 0],
            'cal_parameter_name': 'delta_ccz'
        },
        analysis_options={'outcome': '010'}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_toffoli_default(runner):
    from ..experiments.process_tomography import CircuitTomography

    # 8CX decomposition in Duckering et al. ASPLOS 2021 (cited by 2109.00558)
    circuit = QuantumCircuit(3)
    circuit.t(0)
    circuit.t(1)
    circuit.h(2)
    circuit.cx(0, 1)
    circuit.t(2)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.t(2)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.tdg(2)
    circuit.tdg(1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.tdg(2)
    circuit.cx(1, 2)
    circuit.h(2)

    if 'cx' not in runner.backend.target:
        # Assuming ECR acts on c1->c2 (no layout applied; translator assumes q0 is the control)
        pm = generate_translation_passmanager(runner.backend.operation_names)
        pm.append(ConsolidateRZAngle())
        circuit = pm.run(circuit)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        run_options={'shots': 4000}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_toffoli_bare(runner):
    from ..experiments.process_tomography import CircuitTomography

    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits[1:])

    circuit = QuantumCircuit(3)
    circuit.append(XminusGate(), [1])
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.append(QutritQubitCXGate(), [1, 2])
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.append(XplusGate(), [1])

    # No layout, just translation
    # If CX gate is of reverse (t->c2) type, take it out from the basis gate list here to trigger
    # the decomposition into the compact (non-refocused) form
    basis_gates = runner.backend.basis_gates + [g.gate_name for g in BASIS_GATES]
    if rcr_type == QutritQubitCXType.REVERSE:
        basis_gates.remove('qutrit_qubit_cx')
    pm = generate_translation_passmanager(basis_gates)
    pm.append(ConsolidateRZAngle())
    circuit = pm.run(circuit)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        run_options={'shots': 4000}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_toffoli_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits)
    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        run_options={'shots': 2000, 'rep_delay': runner.backend.configuration().rep_delay_range[1]}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_ccz_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    target_circuit = QuantumCircuit(3)
    target_circuit.ccz(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        experiment_options={'max_circuits': 100},
        run_options={'shots': 2000},
        analysis_options={'target_bootstrap_samples': 100}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_ccz_bc_fullsched(runner):
    from ..experiments.process_tomography import CircuitTomography
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    circuit = circuit_to_pulse_circuit(circuit, runner.backend, runner.calibrations, runner.qubits,
                                       qutrit_transpile_options={'rz_casted_gates': 'all'})
    target_circuit = QuantumCircuit(3)
    target_circuit.ccz(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        experiment_options={'max_circuits': 100},
        run_options={'shots': 2000}
    )

@register_exp
@add_qpt_readout_mitigation
def qpt_cz_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    pm = qutrit_toffoli_translator(runner.backend, runner.calibrations, runner.qubits[1:])
    circuit = QuantumCircuit(2)
    circuit.append(QutritQubitCZGate(), [0, 1])
    circuit = pm.run(circuit)

    target_circuit = QuantumCircuit(2)
    target_circuit.cz(0, 1)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits[1:],
        args={'circuit': circuit, 'target_circuit': target_circuit},
        experiment_options={'max_circuits': 100},
        run_options={'shots': 4000}
    )

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_toffoli(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits)
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': circuit
        }
    )

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_ccz(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': circuit
        }
    )

@register_exp
@add_readout_mitigation(probability=False)
def truthtable_ccz_fullsched(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    circuit = circuit_to_pulse_circuit(circuit, runner.backend, runner.calibrations, runner.qubits,
                                       qutrit_transpile_options={'rz_casted_gates': 'all'})

    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={'circuit': circuit}
    )

@register_exp
@add_readout_mitigation
def phasetable_ccz(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': circuit
        }
    )

@register_exp
@add_readout_mitigation
def phasetable_ccz_fullsched(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    circuit = circuit_to_pulse_circuit(circuit, runner.backend, runner.calibrations, runner.qubits,
                                       qutrit_transpile_options={'rz_casted_gates': 'all'})
    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': circuit
        }
    )
