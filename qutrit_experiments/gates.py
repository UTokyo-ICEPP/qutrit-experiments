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
QUTRIT_COMPOSITE_GATES = []


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
        elif gate_type == GateType.COMPOSITE:
            QUTRIT_COMPOSITE_GATES.append(cls)
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


class RCRGate(QutritGate, gate_name='rcr', gate_type=GateType.COMPOSITE, qutrit=(True, False)):
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

    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        raise NotImplementedError('RCRGate is abstract')

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params=params, label=label)


class RCRTypeXGate(RCRGate, gate_name='rcr2', gate_type=GateType.COMPOSITE, qutrit=(True, False)):
    """Repeated cross resonance gate."""
    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        params = params or {}
        if (cra_params := params.get('cra')) and (crb_params := params.get('crb')):
            cra_gate = CrossResonanceGate(params=cra_params)
            cra_gate.name = 'cra'
            crb_gate = CrossResonanceGate(params=crb_params)
            crb_gate.name = 'crb'
        else:
            cra_gate = crb_gate = CrossResonanceGate(params=params.get('cr'))

        circuit = QuantumCircuit(2)
        circuit.barrier()
        circuit.x(0)
        circuit.x(1)
        circuit.x(1)
        circuit.append(cra_gate, [0, 1])
        circuit.x(0)
        circuit.x(1)
        circuit.x(1)
        circuit.append(crb_gate, [0, 1])
        return circuit


class RCRTypeX12Gate(RCRGate, gate_name='rcr0', gate_type=GateType.COMPOSITE, qutrit=(True, False)):
    """Repeated cross resonance gate."""
    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        params = params or {}
        if (cra_params := params.get('cra')) and (crb_params := params.get('crb')):
            cra_gate = CrossResonanceGate(params=cra_params)
            cra_gate.name = 'cra'
            crb_gate = CrossResonanceGate(params=crb_params)
            crb_gate.name = 'crb'
        else:
            cra_gate = crb_gate = CrossResonanceGate(params=params.get('cr'))
            
        circuit = QuantumCircuit(2)
        circuit.append(cra_gate, [0, 1])
        circuit.append(X12Gate(), [0])
        circuit.x(1)
        circuit.x(1)
        circuit.append(crb_gate, [0, 1])
        circuit.append(X12Gate(), [0])
        circuit.x(1)
        circuit.x(1)
        circuit.barrier()
        return circuit


class QutritQubitCXGate(QutritGate, gate_name='qutrit_qubit_cx', gate_type=GateType.COMPOSITE,
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

    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        raise NotImplementedError('QutritQubitCXGate is abstract')

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params=params, label=label)


class QutritQubitCXTypeXGate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr2',
                             gate_type=GateType.COMPOSITE, qutrit=(True, False)):
    """CX gate with a control qutrit and target qubit."""
    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        params = params or {}
        if (crma_params := params.get('crma')) and (crmb_params := params.get('crmb')):
            crma_gate = CrossResonanceGate(params=crma_params)
            crma_gate.name = 'cra'
            crmb_gate = CrossResonanceGate(params=crmb_params)
            crmb_gate.name = 'crb'
        else:
            crma_gate = crmb_gate = CrossResonanceMinusGate(params=params.get('crm'))
        if (crpa_params := params.get('crpa')) and (crpb_params := params.get('crpb')):
            crpa_gate = CrossResonanceGate(params=crpa_params)
            crpa_gate.name = 'cra'
            crpb_gate = CrossResonanceGate(params=crpb_params)
            crpb_gate.name = 'crb'
        else:
            crpa_gate = crpb_gate = CrossResonanceMinusGate(params=params.get('crp'))

        circuit = QuantumCircuit(2)
        # [Rx]
        circuit.append(CXOffsetRxGate(params.get('rx')), [1])
        # [X+]-[RCR-]
        circuit.append(X12Gate(), [0])
        # Skipping DD here because we have the Rx
        circuit.append(crma_gate, [0, 1])
        circuit.x(0)
        circuit.x(1)
        circuit.x(1)
        circuit.append(crmb_gate, [0, 1])
        # [X+]-[RCR+] x 2
        for _ in range(2):
            circuit.append(X12Gate(), [0])
            circuit.x(1)
            circuit.x(1)
            circuit.append(crpa_gate, [0, 1])
            circuit.x(0)
            circuit.x(1)
            circuit.x(1)
            circuit.append(crpb_gate, [0, 1])
        return circuit


class QutritQubitCXTypeX12Gate(QutritQubitCXGate, gate_name='qutrit_qubit_cx_rcr0',
                               gate_type=GateType.COMPOSITE, qutrit=(True, False)):
    """CX gate with a control qutrit and target qubit."""
    @classmethod
    def decomposition(
        cls,
        params: Optional[dict[str, Any]] = None
    ) -> QuantumCircuit:
        params = params or {}
        if (crma_params := params.get('crma')) and (crmb_params := params.get('crmb')):
            crma_gate = CrossResonanceGate(params=crma_params)
            crma_gate.name = 'cra'
            crmb_gate = CrossResonanceGate(params=crmb_params)
            crmb_gate.name = 'crb'
        else:
            crma_gate = crmb_gate = CrossResonanceMinusGate(params=params.get('crm'))
        if (crpa_params := params.get('crpa')) and (crpb_params := params.get('crpb')):
            crpa_gate = CrossResonanceGate(params=crpa_params)
            crpa_gate.name = 'cra'
            crpb_gate = CrossResonanceGate(params=crpb_params)
            crpb_gate.name = 'crb'
        else:
            crpa_gate = crpb_gate = CrossResonanceMinusGate(params=params.get('crp'))

        circuit = QuantumCircuit(2)
        # [RCR+]-[X+] x 2
        for _ in range(2):
            circuit.append(crpa_gate, [0, 1])
            circuit.append(X12Gate(), [0])
            circuit.x(1)
            circuit.x(1)
            circuit.append(crpb_gate, [0, 1])
            circuit.x(0)
            circuit.x(1)
            circuit.x(1)
        # [RCR-]-[X+]
        circuit.append(crma_gate, [0, 1])
        circuit.append(X12Gate(), [0])
        circuit.x(1)
        circuit.x(1)
        circuit.append(crmb_gate, [0, 1])
        circuit.x(0)
        # Skipping DD here because we have the Rx
        # [Rx]
        circuit.append(CXOffsetRxGate(params.get('rx')), [1])
        return circuit


class CXOffsetRxGate(Gate):
    """Rx gate with a fixed angle to nullify the 0 and 2 block rotations of qutrit-qubit CX."""
    def __init__(
        self,
        angle: Optional[ParameterValueType] = None,
        label: Optional[str] = None
    ):
        if angle is None:
            params = []
        else:
            params = [angle]
        super().__init__('cx_offset_rx', 1, params=params, label=label)


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
