"""Qutrit gates."""

from typing import Optional
from qiskit.circuit import Gate
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
