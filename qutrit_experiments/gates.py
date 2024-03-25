"""Qutrit gates."""

from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.parameterexpression import ParameterValueType

QUTRIT_PULSE_GATES = []
QUTRIT_VIRTUAL_GATES = []
QUTRIT_COMPOSITE_GATES = []


class QutritGate(Gate):
    """Generic qutrit gate."""
    def __init_subclass__(cls, /, gate_name, gate_type, num_qubits, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.gate_name = gate_name
        if gate_type == 'pulse':
            QUTRIT_PULSE_GATES.append(cls)
        elif gate_type == 'virtual':
            QUTRIT_VIRTUAL_GATES.append(cls)
        elif gate_type == 'composite':
            QUTRIT_COMPOSITE_GATES.append(cls)
        cls.num_qubits = num_qubits

    def __init__(self, params, name=None, num_qubits=None, label=None, duration=None, unit='dt'):
        if name is None:
            name = self.gate_name
        if num_qubits is None:
            num_qubits = self.num_qubits
        super().__init__(name, num_qubits, params, label=label, duration=duration, unit=unit)
        

class X12Gate(QutritGate, gate_name='x12', gate_type='pulse', num_qubits=1):
    """The single-qubit X gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__([], label=label)


class SX12Gate(QutritGate, gate_name='sx12', gate_type='pulse', num_qubits=1):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)


class RZ12Gate(QutritGate, gate_name='rz12', gate_type='virtual', num_qubits=1):
    """The RZ gate on EF subspace."""
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__([phi], label=label)


class SetF12Gate(QutritGate, gate_name='set_f12', gate_type='virtual', num_qubits=1):
    """Set the qutrit frequency to a specific value."""
    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__([freq], label=label)


class U12Gate(QutritGate, gate_name='u12', gate_type='composite', num_qubits=1):
    """U gate composed of SX12 and RZ12."""
    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None,
        *,
        duration=None,
        unit="dt",
    ):
        super().__init__([theta, phi, lam], label=label, duration=duration, unit=unit)

    def inverse(self):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return U12Gate(-self.params[0], -self.params[2], -self.params[1])


class CrossResonanceGate(QutritGate, gate_name='cr', gate_type='pulse', num_qubits=2):
    """CR gate with a control qutrit and target qubit."""
    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        block_unitaries: Optional[np.ndarray] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        super().__init__(params, label=label)
        self.block_unitaries = block_unitaries


class CrossResonancePlusGate(CrossResonanceGate, gate_name='crp', gate_type='pulse', num_qubits=2):
    """CR+ gate."""


class CrossResonanceMinusGate(CrossResonanceGate, gate_name='crm', gate_type='pulse', num_qubits=2):
    """CR- gate."""


class QutritQubitCXTypeXGate(QutritGate, gate_name='qutrit_qubit_cx_rcr2', gate_type='composite',
                             num_qubits=2):
    """CX gate with a control qutrit and target qubit."""
    @classmethod
    def decomposition(cls, params) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        # [Rx]
        circuit.rz(np.pi / 2., 1)
        circuit.sx(1)
        circuit.rz(params.get('rx', 0.) + np.pi, 1)
        circuit.sx(1)
        circuit.rz(np.pi / 2., 1)
        # [X+]-[RCR-]
        circuit.append(X12Gate(), [0])
        circuit.append(CrossResonanceMinusGate(params.get('crm')), [0, 1])
        circuit.x(0)
        circuit.append(CrossResonanceMinusGate(params.get('crm')), [0, 1])
        # [X+]-[RCR+] x 2
        for _ in range(2):
            circuit.append(X12Gate(), [0])
            circuit.append(CrossResonancePlusGate(params.get('crp')), [0, 1])
            circuit.x(0)
            circuit.append(CrossResonancePlusGate(params.get('crp')), [0, 1])
        return circuit

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)


class QutritQubitCXTypeX12Gate(QutritGate, gate_name='qutrit_qubit_cx_rcr0', gate_type='composite',
                               num_qubits=2):
    """CX gate with a control qutrit and target qubit."""
    @classmethod
    def decomposition(cls, params) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        # [RCR+]-[X+] x 2
        for _ in range(2):
            circuit.append(CrossResonancePlusGate(params.get('crp')), [0, 1])
            circuit.append(X12Gate(), [0])
            circuit.append(CrossResonancePlusGate(params.get('crp')), [0, 1])
            circuit.x(0)
        # [RCR-]-[X+]
        circuit.append(CrossResonanceMinusGate(params.get('crm')), [0, 1])
        circuit.append(X12Gate(), [0])
        circuit.append(CrossResonanceMinusGate(params.get('crm')), [0, 1])
        circuit.x(0)
        # [Rx]
        circuit.rz(np.pi / 2., 1)
        circuit.sx(1)
        circuit.rz(params.get('rx', 0.) + np.pi, 1)
        circuit.sx(1)
        circuit.rz(np.pi / 2., 1)
        return circuit

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)


q = QuantumRegister(1, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u3_qasm_def = QuantumCircuit(q, global_phase=(lam + phi - np.pi) / 2)
u3_qasm_def.append(RZ12Gate(lam), [0])
u3_qasm_def.append(SX12Gate(), [0])
u3_qasm_def.append(RZ12Gate(theta + np.pi), [0])
u3_qasm_def.append(SX12Gate(), [0])
u3_qasm_def.append(RZ12Gate(phi + 3 * np.pi), [0])
sel.add_equivalence(U12Gate(theta, phi, lam), u3_qasm_def)
