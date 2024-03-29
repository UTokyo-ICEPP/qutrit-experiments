"""Qutrit gates."""

from collections.abc import Sequence
from enum import Enum, IntEnum, auto
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.parameterexpression import ParameterValueType

QUTRIT_PULSE_GATES = []
QUTRIT_VIRTUAL_GATES = []

class GateType(Enum):
    PULSE = auto()
    VIRTUAL = auto()
    COMPOSITE = auto()


class QutritGate(Gate):
    """Generic qutrit gate."""
    def __init_subclass__(cls, /, gate_name, gate_type, qutrit=(True,), **kwargs):
        super().__init_subclass__(**kwargs)
        cls.gate_name = gate_name
        if gate_type == GateType.PULSE:
            QUTRIT_PULSE_GATES.append(cls)
        elif gate_type == GateType.VIRTUAL:
            QUTRIT_VIRTUAL_GATES.append(cls)
        cls.qutrit = qutrit

    def __init__(self, params, name=None, num_qubits=None, label=None, duration=None, unit='dt'):
        if name is None:
            name = self.gate_name
        if num_qubits is None:
            num_qubits = len(self.qutrit)
        super().__init__(name, num_qubits, params, label=label, duration=duration, unit=unit)


class X12Gate(QutritGate, gate_name='x12', gate_type=GateType.PULSE):
    """The single-qubit X gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__([], label=label)


class SX12Gate(QutritGate, gate_name='sx12', gate_type=GateType.PULSE):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)


class RZ12Gate(QutritGate, gate_name='rz12', gate_type=GateType.VIRTUAL):
    """The RZ gate on EF subspace."""
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__([phi], label=label)


class SetF12Gate(QutritGate, gate_name='set_f12', gate_type=GateType.VIRTUAL):
    """Set the qutrit frequency to a specific value."""
    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__([freq], label=label)


class U12Gate(QutritGate, gate_name='u12', gate_type=GateType.COMPOSITE):
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


class CrossResonanceGate(QutritGate, gate_name='cr', gate_type=GateType.PULSE, qutrit=(True, False)):
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


class CrossResonancePlusGate(CrossResonanceGate, gate_name='crp', gate_type=GateType.PULSE,
                             qutrit=(True, False)):
    """CR+ gate."""


class CrossResonanceMinusGate(CrossResonanceGate, gate_name='crm', gate_type=GateType.PULSE,
                              qutrit=(True, False)):
    """CR- gate."""


class RCRGate(QutritGate, gate_name='rcr', gate_type=GateType.PULSE, qutrit=(True, False)):
    """Repeated cross resonance gate."""
    TYPE_X = 2 # CRCR angle = 2 * (θ_0 + θ_1 - 2*θ_2)
    TYPE_X12 = 0 # CRCR angle = 2 * (θ_1 + θ_2 - 2*θ_0)

    @classmethod
    def of_type(cls, rcr_type: int) -> type['RCRGate']:
        match rcr_type:
            case cls.TYPE_X:
                return RCRTypeXGate
            case cls.TYPE_X12:
                return RCRTypeX12Gate

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params=params, label=label)


class RCRTypeXGate(RCRGate, gate_name='rcr2', gate_type=GateType.PULSE, qutrit=(True, False)):
    """Repeated cross resonance gate."""


class RCRTypeX12Gate(RCRGate, gate_name='rcr0', gate_type=GateType.PULSE, qutrit=(True, False)):
    """Repeated cross resonance gate."""


class QutritQubitCXGate(QutritGate, gate_name='qutrit_qubit_cx', gate_type=GateType.PULSE,
                        qutrit=(True, False)):
    """CX gate with a control qutrit and target qubit."""
    TYPE_X = 2 # CRCR angle = 2 * (θ_0 + θ_1 - 2*θ_2)
    TYPE_X12 = 0 # CRCR angle = 2 * (θ_1 + θ_2 - 2*θ_0)

    @classmethod
    def of_type(cls, rcr_type: int) -> type['QutritQubitCXGate']:
        match rcr_type:
            case cls.TYPE_X:
                return QutritQubitCXTypeXGate
            case cls.TYPE_X12:
                return QutritQubitCXTypeX12Gate

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params=params, label=label)


class QutritQubitCXTypeXGate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr2',
                             gate_type=GateType.PULSE, qutrit=(True, False)):
    """CX gate with a control qutrit and target qubit."""


class QutritQubitCXTypeX12Gate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr0',
                               gate_type=GateType.PULSE, qutrit=(True, False)):
    """CX gate with a control qutrit and target qubit."""


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
