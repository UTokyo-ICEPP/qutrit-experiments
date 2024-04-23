# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from qiskit import QuantumCircuit
from qiskit_experiments.data_processing import DataProcessor

from ..experiment_config import ExperimentConfig, register_exp
from ..gates import (QutritCCZGate, QutritQubitCXType, QutritQubitCXGate, QutritQubitCZGate,
                     XminusGate, XplusGate)
from ..transpilation.layout_and_translation import generate_translation_passmanager
from ..transpilation.qutrit_transpiler import BASIS_GATES
from ..transpilation.rz import ConsolidateRZAngle
from ..util.qutrit_toffoli import qutrit_toffoli_circuit, qutrit_toffoli_translator
from .common import add_readout_mitigation


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
def toffoli_qpt_default(runner):
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
@add_readout_mitigation
def toffoli_qpt_bare(runner):
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
@add_readout_mitigation
def toffoli_qpt_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits)
    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        run_options={'shots': 4000}
    )

@register_exp
@add_readout_mitigation
def ccz_qpt_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    target_circuit = QuantumCircuit(3)
    target_circuit.ccz(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit},
        run_options={'shots': 4000}
    )

@register_exp
@add_readout_mitigation
def toffoli_truth_table(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits)
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': circuit
        },
        # next line required for add_readout_mitigation to work properly (we don't want Probability)
        analysis_options={'data_processor': DataProcessor('counts', [])}
    )

@register_exp
@add_readout_mitigation
def ccz_truth_table(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    circuit = qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits,
                                     gate=QutritCCZGate())
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': circuit
        },
        # next line required for add_readout_mitigation to work properly (we don't want Probability)
        analysis_options={'data_processor': DataProcessor('counts', [])}
    )

@register_exp
@add_readout_mitigation
def ccz_phase_table(runner):
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
def cz_phase_table(runner):
    from qutrit_experiments.experiments.phase_table import PhaseTable
    pm = qutrit_toffoli_translator(runner.backend, runner.calibrations, runner.qubits[1:])
    circuit = QuantumCircuit(2)
    circuit.append(QutritQubitCZGate(), [0, 1])
    circuit = pm.run(circuit)

    return ExperimentConfig(
        PhaseTable,
        runner.qubits,
        args={
            'circuit': circuit
        }
    )
