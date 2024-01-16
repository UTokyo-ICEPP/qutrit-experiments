"""Qutrit gates."""

from typing import Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.parameterexpression import ParameterValueType

QUTRIT_PULSE_GATES = []
QUTRIT_VIRTUAL_GATES = []


class QutritGate(Gate):
    """Generic qutrit gate."""
    def __init_subclass__(cls, /, gate_name, gate_type, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.gate_name = gate_name
        if gate_type == 'pulse':
            QUTRIT_PULSE_GATES.append(cls)
        elif gate_type == 'virtual':
            QUTRIT_VIRTUAL_GATES.append(cls)


class X12Gate(QutritGate, gate_name='x12', gate_type='pulse'):
    """The single-qubit X gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__(self.gate_name, 1, [], label=label)


class SX12Gate(QutritGate, gate_name='sx12', gate_type='pulse'):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__(self.gate_name, 1, [], label=label)


class RZ12Gate(QutritGate, gate_name='rz12', gate_type='virtual'):
    """The RZ gate on EF subspace."""
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__(self.gate_name, 1, [phi], label=label)


class SetF12Gate(QutritGate, gate_name='set_f12', gate_type='virtual'):
    """Set the qutrit frequency to a specific value."""
    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__(self.gate_name, 1, [freq], label=label)


class U12Gate(QutritGate, gate_name='u12', gate_type='composite'):
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
        super().__init__('u12', 1, [theta, phi, lam], label=label, duration=duration, unit=unit)

    def inverse(self):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return U12Gate(-self.params[0], -self.params[2], -self.params[1])


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
