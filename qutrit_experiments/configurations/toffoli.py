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
    qubits = tuple(runner.qubits)

    cx_circuit = QuantumCircuit(2)
    cx_circuit.cx(0, 1)
    if 'cx' not in runner.backend.target:
        # Assuming ECR acts on c1->c2 (no layout applied; translator assumes q0 is the control)
        pm = generate_translation_passmanager(runner.backend.operation_names)
        pm.append(ConsolidateRZAngle())
        cx_circuit = pm.run(cx_circuit)

    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits[1:])

    circuit = QuantumCircuit(3)
    circuit.append(X12Gate(label='qutrit_circuit_start'), [1])
    circuit.x(1)
    circuit.delay(Parameter('qutrit_refocusing_delay'))
    circuit.append(X12Gate(), [1])
    circuit.x(1)
    circuit.compose(cx_circuit, [0, 1], inplace=True)
    circuit.append(QutritQubitCXGate.of_type(rcr_type)(), [1, 2])
    circuit.compose(cx_circuit, [0, 1], inplace=True)
    circuit.append(X12Gate(), [1])
    circuit.x(1)

    instruction_durations = make_instruction_durations(runner.backend, runner.calibrations,
                                                       qubits=qubits)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit}
    )
