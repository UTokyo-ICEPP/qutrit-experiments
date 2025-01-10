"""Qutrit gates."""

from collections.abc import Sequence
from enum import Enum, IntEnum, auto
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Barrier, Gate, Parameter
from qiskit.circuit.library import ECRGate, HGate, RZGate, SXGate, XGate, ZGate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
# from qiskit.circuit.parameterexpression import ParameterValueType
#  For some reason Pylance fails to recognize imported ParameterValueType as a valid type alias so
#  I'm redefining it here
from qiskit.circuit.parameterexpression import ParameterExpression
ParameterValueType = Union[ParameterExpression, float]


class GateType(Enum):
    """Gate type descriptors."""
    PULSE = auto()
    VIRTUAL = auto()
    COMPOSITE = auto()


class QutritGate(Gate):
    """Qutrit gate base class."""
    gate_type = None

    def __init__(
        self,
        name: str,
        as_qutrit: tuple[bool, ...],
        params: list[ParameterValueType],
        label: Optional[str] = None,
        duration: Optional[int] = None,
        unit: str = 'dt'
    ):
        super().__init__(name, len(as_qutrit), params, label=label, duration=duration, unit=unit)
        self.as_qutrit = as_qutrit


class QutritPulseGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.PULSE


class QutritVirtualGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.VIRTUAL


class QutritCompositeGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.COMPOSITE


# pylint: disable=too-few-public-methods
class StaticProperties:
    """Mixin for qutrit gates with properties defined at class level. Copies of the properties will
    be set on instances through __init__()."""
    def __init__(self, params, *args, label=None, duration=None, unit='dt', **kwargs):
        if params is None:
            params = []
        else:
            params = list(params)
        # pylint: disable=no-member
        super().__init__(self.__class__.gate_name, self.__class__.as_qutrit, params, *args,
                         label=label, duration=duration, unit=unit, **kwargs)


class OneQutritPulseGate(StaticProperties, QutritPulseGate):
    """Single qutrit pulse gate base class."""
    as_qutrit = (True,)


class OneQutritVirtualGate(StaticProperties, QutritVirtualGate):
    """Single qutrit virtual gate base class."""
    as_qutrit = (True,)


class OneQutritCompositeGate(StaticProperties, QutritCompositeGate):
    """Single qutrit composite gate base class."""
    as_qutrit = (True,)


class QutritQubitCompositeGate(StaticProperties, QutritCompositeGate):
    """Qutrit-qubit composite gate base class."""
    as_qutrit = (True, False)


class X12Gate(OneQutritPulseGate):
    """The single-qubit X gate on EF subspace."""
    gate_name = 'x12'

    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__([], label=label)


class SX12Gate(OneQutritPulseGate):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    gate_name = 'sx12'

    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)


class RZ12Gate(OneQutritVirtualGate):
    """The RZ gate on EF subspace."""
    gate_name = 'rz12'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__([phi], label=label)


class SetF12Gate(OneQutritVirtualGate):
    """Set the qutrit frequency to a specific value."""
    gate_name = 'set_f12'

    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__([freq], label=label)


class U12Gate(OneQutritCompositeGate):
    """U gate composed of SX12 and RZ12."""
    gate_name = 'u12'

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


class XplusGate(OneQutritCompositeGate):
    """X+ gate."""
    gate_name = 'xplus'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)


class XminusGate(OneQutritCompositeGate):
    """X- gate."""
    gate_name = 'xminus'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)


class X02Gate(OneQutritCompositeGate):
    """X02 gate."""
    gate_name = 'x02'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)


class P0Gate(OneQutritCompositeGate):
    """P0 gate."""
    gate_name = 'p0'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P0 gate."""
        super().__init__([phi], label=label)


class P1Gate(OneQutritCompositeGate):
    """P1 gate."""
    gate_name = 'p1'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P1 gate."""
        super().__init__([phi], label=label)


class P2Gate(OneQutritCompositeGate):
    """P2 gate."""
    gate_name = 'p2'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P2 gate."""
        super().__init__([phi], label=label)


class QGate(OneQutritCompositeGate):
    """Q gate."""
    gate_name = 'q'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new Q gate."""
        super().__init__([phi], label=label)


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


class QutritQubitCXType(IntEnum):
    """Type of qutrit-qubit generalized CX gate."""
    X = 2
    X12 = 0
    REVERSE = -1
    CRCR = -2


class RCRGate(Gate):
    """Repeated cross resonance gate."""
    gate_name = 'rcr'
    gate_type = GateType.COMPOSITE

    def __init__(
        self,
        params: list[ParameterValueType],
        label: Optional[str] = None,
        duration: Optional[int] = None,
        unit: str = 'dt'
    ):
        super().__init__('rcr', 2, params, label=label, duration=duration, unit=unit)


class CRCRGate(QutritQubitCompositeGate):
    """Cycled RCR gate.

    CRCR angles:
    Type X(2) -> 2 * (θ_0 + θ_1 - 2*θ_2)
    Type X12(0) -> 2 * (θ_1 + θ_2 - 2*θ_0)
    """
    gate_name = 'crcr'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritQubitCXGate(QutritQubitCompositeGate):
    """CX gate with a control qutrit and target qubit."""
    gate_name = 'qutrit_qubit_cx'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritQubitCZGate(QutritQubitCompositeGate):
    """CZ gate with a control qutrit and target qubit."""
    gate_name = 'qutrit_qubit_cz'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritMCGate(QutritCompositeGate):
    """Multi-controlled gate using qutrits."""
    def __init__(
        self,
        name: str,
        num_controls: int,
        target_gate: type[Gate],
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        as_qutrit = (False,) + (True,) * (num_controls - 1) + (False,)
        super().__init__(name, as_qutrit, params, label=label)
        self.target_gate = target_gate


class QutritCCXGate(QutritMCGate):
    """Toffoli gate using qutrits."""
    gate_name = 'qutrit_toffoli'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(self.gate_name, 2, XGate, params=params, label=label)


class QutritCCZGate(QutritMCGate):
    """CCZ gate using qutrits."""
    gate_name = 'qutrit_ccz'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(self.gate_name, 2, ZGate, params=params, label=label)


q = QuantumRegister(1, 'q')
_theta = Parameter('theta')
_phi = Parameter('phi')
_lam = Parameter('lam')
qasm_def = QuantumCircuit(q, global_phase=(_lam + _phi - np.pi) / 2)
qasm_def.append(RZ12Gate(_lam), [0])
qasm_def.append(SX12Gate(), [0])
qasm_def.append(RZ12Gate(_theta + np.pi), [0])
qasm_def.append(SX12Gate(), [0])
qasm_def.append(RZ12Gate(_phi + 3 * np.pi), [0])
sel.add_equivalence(U12Gate(_theta, _phi, _lam), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
sel.add_equivalence(XplusGate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
sel.add_equivalence(XminusGate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
sel.add_equivalence(X02Gate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
sel.add_equivalence(X02Gate(), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(-_phi * 4. / 3., 0)
qasm_def.append(RZ12Gate(-_phi * 2. / 3.), [0])
sel.add_equivalence(P0Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2. / 3., 0)
qasm_def.append(RZ12Gate(-_phi * 2. / 3.), [0])
sel.add_equivalence(P1Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2. / 3., 0)
qasm_def.append(RZ12Gate(_phi * 4. / 3.), [0])
sel.add_equivalence(P2Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2., 0)
qasm_def.append(RZ12Gate(_phi * 2.), [0])
sel.add_equivalence(QGate(_phi), qasm_def)

# Compact version of reverse CX
q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (HGate(), [1]),
    (QutritQubitCZGate(), [0, 1]),
    (HGate(), [1])
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritQubitCXGate(), qasm_def)

# Compact version of reverse CZ
q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (XplusGate(), [0]),
    (XGate(), [1]),
    (ECRGate(), [1, 0]),
    (XGate(), [1]),
    (ECRGate(), [1, 0]),
    (X12Gate(), [0]),
    (RZGate(np.pi / 3.), [0]),
    (RZ12Gate(-np.pi / 3.), [0]),
    (RZGate(np.pi), [1])
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritQubitCZGate(), qasm_def)

q = QuantumRegister(3, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (Barrier(3), q),
    (XminusGate(label='qutrit_toffoli_begin'), q[1:2]),
    (Barrier(2), q[:2]),
    (RZGate(-np.pi / 2.), q[:1]),
    (ECRGate(), q[:2]),
    (XGate(), q[:1]),
    (RZGate(-np.pi), q[1:2]),
    (SXGate(), q[1:2]),
    (RZGate(-np.pi), q[1:2]),
    (Barrier(3), q),
    (QutritQubitCXGate(), q[1:]),
    (Barrier(3), q),
    (XGate(), q[:1]),
    (ECRGate(), q[:2]),
    (RZGate(np.pi / 2.), q[:1]),
    (RZGate(-np.pi / 3.), q[1:2]),
    (RZ12Gate(-2. * np.pi / 3.), q[1:2]),
    (SXGate(), q[1:2]),
    (Barrier(2), q[:2]),
    (XplusGate(label='qutrit_toffoli_end'), q[1:2]),
    (Barrier(3), q)
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritCCXGate(), qasm_def)

q = QuantumRegister(3, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (Barrier(3), q),
    (XminusGate(label='qutrit_toffoli_begin'), q[1:2]),
    (Barrier(2), q[:2]),
    (RZGate(-np.pi / 2.), q[:1]),
    (ECRGate(), q[:2]),
    (XGate(), q[:1]),
    (RZGate(-np.pi), q[1:2]),
    (SXGate(), q[1:2]),
    (RZGate(-np.pi), q[1:2]),
    (Barrier(3), q),
    (QutritQubitCZGate(), q[1:]),
    (Barrier(3), q),
    (XGate(), q[:1]),
    (ECRGate(), q[:2]),
    (RZGate(np.pi / 2.), q[:1]),
    (RZGate(-np.pi / 3.), q[1:2]),
    (RZ12Gate(-2. * np.pi / 3.), q[1:2]),
    (SXGate(), q[1:2]),
    (Barrier(2), q[:2]),
    (XplusGate(label='qutrit_toffoli_end'), q[1:2]),
    (Barrier(3), q)
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritCCZGate(), qasm_def)
