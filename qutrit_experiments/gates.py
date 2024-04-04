"""Qutrit gates."""

from collections.abc import Sequence
from enum import Enum, auto
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
#from qiskit.circuit.parameterexpression import ParameterValueType
# For some reason Pylance fails to recognize imported ParameterValueType as a valid type alias so
# I'm redefining it here
from qiskit.circuit.parameterexpression import ParameterExpression
ParameterValueType = Union[ParameterExpression, float]


class GateType(Enum):
    PULSE = auto()
    VIRTUAL = auto()
    COMPOSITE = auto()


class QutritGate(Gate):
    """Qutrit gate base class."""
    gate_name = None
    gate_type = None
    as_qutrit = None
    num_qubits = None

    def __init_subclass__(cls, /, gate_name, gate_type, as_qutrit=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.gate_name = gate_name
        cls.gate_type = gate_type
        # Get as_qutrit from the direct parent if None
        cls.as_qutrit = as_qutrit or cls.mro()[1].as_qutrit
        if cls.as_qutrit is not None:
            cls.num_qubits = len(cls.as_qutrit)

    def __init__(self, params, name=None, label=None, duration=None, unit='dt'):
        if name is None:
            name = self.gate_name
        super().__init__(name, self.num_qubits, params, label=label, duration=duration, unit=unit)


class QutritPulseGateBase(QutritGate, gate_name='', gate_type=GateType.PULSE):
    """Pulse gate base class."""
    def __init_subclass__(cls, /, gate_name, as_qutrit=None, **kwargs):
        super().__init_subclass__(gate_name=gate_name, gate_type=GateType.PULSE,
                                  as_qutrit=as_qutrit, **kwargs)


class QutritVirtualGateBase(QutritGate, gate_name='', gate_type=GateType.VIRTUAL):
    """Pulse gate base class."""
    def __init_subclass__(cls, /, gate_name, as_qutrit=None, **kwargs):
        super().__init_subclass__(gate_name=gate_name, gate_type=GateType.VIRTUAL,
                                  as_qutrit=as_qutrit, **kwargs)


class QutritCompositeGateBase(QutritGate, gate_name='', gate_type=GateType.COMPOSITE):
    """Pulse gate base class."""
    def __init_subclass__(cls, /, gate_name, as_qutrit=None, **kwargs):
        super().__init_subclass__(gate_name=gate_name, gate_type=GateType.COMPOSITE,
                                  as_qutrit=as_qutrit, **kwargs)


class QutritPulseGate(QutritPulseGateBase, gate_name='pulse'):
    """Per-instance defined qutrit gate."""
    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list[ParameterValueType],
        label: Optional[str] = None
    ):
        super().__init__(params, name=name, label=label)
        self.num_qubits = num_qubits


class QutritCompositeGate(QutritCompositeGateBase, gate_name='composite'):
    """Per-instance defined qutrit gate."""
    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list[ParameterValueType],
        label: Optional[str] = None
    ):
        super().__init__(params, name=name, label=label)
        self.num_qubits = num_qubits


class OneQutritPulseGateBase(QutritPulseGateBase, gate_name='', as_qutrit=(True,)):
    """Single qutrit pulse gate base class."""


class OneQutritVirtualGateBase(QutritVirtualGateBase, gate_name='', as_qutrit=(True,)):
    """Single qutrit virtual gate base class."""


class OneQutritCompositeGateBase(QutritCompositeGateBase, gate_name='', as_qutrit=(True,)):
    """Single qutrit composite gate base class."""


class QutritQubitCompositeGateBase(QutritCompositeGateBase, gate_name='', as_qutrit=(True, False)):
    """Qutrit-qubit composite gate base class."""


class X12Gate(OneQutritPulseGateBase, gate_name='x12'):
    """The single-qubit X gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__([], label=label)


class SX12Gate(OneQutritPulseGateBase, gate_name='sx12'):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)


class RZ12Gate(OneQutritVirtualGateBase, gate_name='rz12'):
    """The RZ gate on EF subspace."""
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__([phi], label=label)


class SetF12Gate(OneQutritVirtualGateBase, gate_name='set_f12'):
    """Set the qutrit frequency to a specific value."""
    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__([freq], label=label)


class U12Gate(OneQutritCompositeGateBase, gate_name='u12'):
    """U gate composed of SX12 and RZ12."""
    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__([theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return U12Gate(-self.params[0], -self.params[2], -self.params[1])


class XplusGate(OneQutritCompositeGateBase, gate_name='xplus'):
    """X+ gate."""
    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)


class XminusGate(OneQutritCompositeGateBase, gate_name='xminus'):
    """X- gate."""
    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)


class CrossResonanceGate(Gate):
    """CR gate with a control qutrit and target qubit."""
    gate_name = 'cr'
    gate_type = GateType.PULSE

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
        super().__init__(self.gate_name, 2, params, label=label)
        self.block_unitaries = block_unitaries


class CrossResonancePlusGate(CrossResonanceGate):
    """CR+ gate."""
    gate_name = 'crp'


class CrossResonanceMinusGate(CrossResonanceGate):
    """CR- gate."""
    gate_name = 'crm'


class RCRGate(Gate):
    """Repeated cross resonance gate."""
    TYPE_X = 2 # CRCR angle = 2 * (θ_0 + θ_1 - 2*θ_2)
    TYPE_X12 = 0 # CRCR angle = 2 * (θ_1 + θ_2 - 2*θ_0)

    gate_name = 'rcr'
    gate_type = GateType.COMPOSITE

    @classmethod
    def of_type(cls, rcr_type: int) -> type['RCRGate']:
        match rcr_type:
            case cls.TYPE_X:
                return RCRTypeXGate
            case cls.TYPE_X12:
                return RCRTypeX12Gate


class RCRTypeXGate(RCRGate):
    """Repeated cross resonance gate."""
    gate_name = 'rcr2'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        super().__init__(self.gate_name, 2, params=params, label=label)


class RCRTypeX12Gate(QutritQubitCompositeGateBase, RCRGate, gate_name='rcr0'):
    """Repeated cross resonance gate."""
    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        super().__init__(params=params, label=label)


class CRCRGate(QutritQubitCompositeGateBase, gate_name='crcr'):
    """Cycled RCR gate.

    CRCR angles:
    TYPE_X(2) -> 2 * (θ_0 + θ_1 - 2*θ_2)
    TYPE_X12(0) -> 2 * (θ_1 + θ_2 - 2*θ_0)
    """
    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params=params, label=label)

    @classmethod
    def of_type(cls, rcr_type: int) -> type['CRCRGate']:
        match rcr_type:
            case RCRGate.TYPE_X:
                return CRCRTypeXGate
            case RCRGate.TYPE_X12:
                return CRCRTypeX12Gate


class CRCRTypeXGate(CRCRGate, gate_name='crcr2'):
    """Cycled RCR gate."""


class CRCRTypeX12Gate(CRCRGate, gate_name='crcr0'):
    """Cycled RCR gate."""


class QutritQubitCXGate(QutritQubitCompositeGateBase, gate_name='qutrit_qubit_cx'):
    """CX gate with a control qutrit and target qubit."""
    @classmethod
    def of_type(cls, rcr_type: int) -> type['QutritQubitCXGate']:
        match rcr_type:
            case RCRGate.TYPE_X:
                return QutritQubitCXTypeXGate
            case RCRGate.TYPE_X12:
                return QutritQubitCXTypeX12Gate

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        super().__init__(params=params, label=label)


class QutritQubitCXTypeXGate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr2'):
    """CX gate with a control qutrit and target qubit."""


class QutritQubitCXTypeX12Gate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr0'):
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

q = QuantumRegister(1, "q")
u3_qasm_def = QuantumCircuit(q)
u3_qasm_def.append(X12Gate(), [0])
u3_qasm_def.x(0)
sel.add_equivalence(XplusGate(), u3_qasm_def)

q = QuantumRegister(1, "q")
u3_qasm_def = QuantumCircuit(q)
u3_qasm_def.x(0)
u3_qasm_def.append(X12Gate(), [0])
sel.add_equivalence(XminusGate(), u3_qasm_def)
