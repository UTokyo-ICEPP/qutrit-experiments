# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from ..experiment_config import ExperimentConfig, register_exp
from ..gates import QutritQubitCXGate, X12Gate, XminusGate, XplusGate
from ..transpilation.layout_and_translation import generate_translation_passmanager
from ..transpilation.qutrit_transpiler import BASIS_GATES, make_instruction_durations
from ..transpilation.rz import ConsolidateRZAngle
from ..util.qutrit_toffoli import qutrit_toffoli_circuit
from .common import add_readout_mitigation


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
    # CX gate is in the list of basis gates and will be attached a calibration in
    # the qutrit circuits transpiler
    circuit.append(QutritQubitCXGate.of_type(rcr_type)(), [1, 2])
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.append(XplusGate(), [1])

    # No layout, just translation
    basis_gates = runner.backend.basis_gates + [g.gate_name for g in BASIS_GATES]
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
def toffoli_truth_table(runner):
    from qutrit_experiments.experiments.truth_table import TruthTable
    return ExperimentConfig(
        TruthTable,
        runner.qubits,
        args={
            'circuit': qutrit_toffoli_circuit(runner.backend, runner.calibrations, runner.qubits)
        }
    )
