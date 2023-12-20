"""Qutrit gates."""

from typing import Optional
from qiskit.circuit import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


class X12Gate(Gate):
    """The single-qubit X gate on EF subspace."""
    def __init__(self, angle: Optional[ParameterValueType] = None, *, label: Optional[str] = None):
        """Create new X12 gate."""
        if angle is None:
            super().__init__('x12', 1, [], label=label)
        else:
            super().__init__('x12', 1, [angle], label=label)


class SX12Gate(Gate):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, angle: Optional[ParameterValueType] = None, *, label: Optional[str] = None):
        """Create new SX12 gate."""
        if angle is None:
            super().__init__('sx12', 1, [], label=label)
        else:
            super().__init__('sx12', 1, [angle], label=label)


class RZ12Gate(Gate):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__('rz12', 1, [phi], label=label)


class SetF12Gate(Gate):
    """Set the qutrit frequency to a specific value."""
    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__('set_f12', 1, [freq], label=label)
