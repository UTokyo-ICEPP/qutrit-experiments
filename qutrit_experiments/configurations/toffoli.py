# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from ..experiment_config import ExperimentConfig, register_exp
from ..experiments.qutrit_qubit_cx.util import make_crcr_circuit
from ..gates import QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES, RZ12Gate, X12Gate
from ..transpilation.layout_and_translation import generate_translation_passmanager
from ..transpilation.qutrit_transpiler import make_instruction_durations
from ..transpilation.rz import ConsolidateRZAngle
from .common import add_readout_mitigation


@register_exp
@add_readout_mitigation
def toffoli_qpt_default(runner):
    from ..experiments.process_tomography import CircuitTomography

    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)

    return ExperimentConfig(
        CircuitTomography,
        runner.qubits,
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

    c2t = tuple(runner.qubits[1:])
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
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit}
    )

@register_exp
@add_readout_mitigation
def toffoli_qpt_bc(runner):
    from ..experiments.process_tomography import CircuitTomography
    qubits = tuple(runner.qubits)

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
        runner.qubits,
        args={'circuit': circuit, 'target_circuit': target_circuit}
    )
