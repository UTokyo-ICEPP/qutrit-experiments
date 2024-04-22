"""Transformation utilities."""
from collections.abc import Sequence
import logging
from typing import Optional, Union
import numpy as np
import scipy
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import ECRGate, RZGate, SXGate, XGate
from qiskit.providers import Backend
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.exceptions import CalibrationError

from ..calibrations import get_qutrit_freq_shift
from ..gates import QutritQubitCXGate, RZ12Gate, SX12Gate, X12Gate

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


def schedule_to_block(schedule: Schedule):
    """Convert a Schedule to ScheduleBlock. Am I sure this doesn't exist in Qiskit proper?"""
    # Convert to ScheduleBlock
    with pulse.build(name=schedule.name) as schedule_block:
        pulse.call(schedule)
    return schedule_block


def symbolic_pulse_to_waveform(schedule_block: ScheduleBlock):
    """Replace custom pulses to waveforms in place."""
    # .blocks returns a new container so we don't need to wrap the result further
    for block in schedule_block.blocks:
        if isinstance((inst := block), pulse.Play) and isinstance(inst.pulse, pulse.SymbolicPulse):
            # Check if the pulse is known by the backend
            try:
                ParametricPulseShapes(inst.pulse.pulse_type)
            except ValueError:
                waveform_inst = pulse.Play(inst.pulse.get_waveform(), inst.channel, name=inst.name)
                schedule_block.replace(inst, waveform_inst, inplace=True)
        elif isinstance(block, ScheduleBlock):
            symbolic_pulse_to_waveform(block)


def circuit_to_matrix(
    circuit: QuantumCircuit,
    qutrits: Sequence[int],
    qubit_ids: Optional[Sequence[int]] = None
) -> np.ndarray:
    qubit_index = {}
    if qubit_ids:
        num_qubits = len(qubit_ids)
        qubit_index = {circuit.qubits[idx]: iq for iq, idx in enumerate(qubit_ids)}
    else:
        num_qubits = circuit.num_qubits
        qubit_index = {qubit: iq for iq, qubit in enumerate(circuit.qubits)}

    dims = tuple(3 if iq in qutrits else 2 for iq in range(num_qubits))
    matrix = np.eye(np.prod(dims), dtype=complex).reshape(dims + dims)

    for inst in circuit.data:
        if isinstance(inst.operation, (Barrier, Delay)):
            continue
        if inst.operation.name in ('dd_left', 'dd_right'):
            continue

        qubits = tuple(qubit_index[q] for q in inst.qubits)
        if len(qubits) == 1:
            dim = dims[qubits[0]]
            in_axes = (1,)

            if isinstance(inst.operation, SXGate):
                if dim == 2:
                    gate = np.array([[1.+1.j, 1.-1.j], [1.-1.j, 1.+1.j]]) / 2.
                else:
                    gate = np.array([[1.+1.j, 1.-1.j, 0.], [1.-1.j, 1.+1.j, 0.], [0., 0., 2.]]) / 2.
            elif isinstance(inst.operation, XGate):
                if dim == 2:
                    gate = np.array([[0., 1.], [1., 0.]], dtype=complex)
                else:
                    gate = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=complex)
            elif isinstance(inst.operation, RZGate):
                phase = inst.operation.params[0]
                diag = [np.exp(-0.5j * phase), np.exp(0.5j * phase)]
                if dim == 3:
                    diag.append(1.)
                gate = np.diagflat(diag)
            elif isinstance(inst.operation, SX12Gate):
                gate = np.array([[2., 0., 0.], [0., 1.+1.j, 1.-1.j], [0., 1.-1.j, 1.+1.j]],
                                dtype=complex) / 2.
            elif isinstance(inst.operation, X12Gate):
                gate = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=complex)
            elif isinstance(inst.operation, RZ12Gate):
                phase = inst.operation.params[0]
                diag = [1., np.exp(-0.5j * phase), np.exp(0.5j * phase)]
                gate = np.diagflat(diag)
            else:
                raise RuntimeError(f'Unhandled instruction {inst.operation}')
        elif len(qubits) == 2:
            dim = tuple(dims[iq] for iq in qubits)
            in_axes = (2, 3)

            if isinstance(inst.operation, ECRGate):
                if dim == (2, 2):
                    gate = np.array(
                        [[[[0., 0.], [1., 1.j]],
                          [[0., 0.], [1.j, 1.]]],
                         [[[1., -1.j], [0., 0.]],
                          [[-1.j, 1.], [0., 0.]]]]
                    ) / np.sqrt(2.)
                elif dim == (2, 3):
                    gate = np.array(
                        [[[[0., 0., 0.], [1., 1.j, 0.]],
                          [[0., 0., 0.], [1.j, 1., 0.]],
                          [[0., 0., 0.], [0., 0., np.sqrt(2.)]]],
                         [[[1., -1.j, 0.], [0., 0., 0.]],
                          [[-1.j, 1., 0.], [0., 0., 0.]],
                          [[0., 0., np.sqrt(2.)], [0., 0., 0.]]]]
                    ) / np.sqrt(2.)
                else:
                    raise RuntimeError(f'Unhandled dimension for ECRGate {dim}')
            else:
                raise RuntimeError(f'Unhandled instruction {inst.operation}')
        else:
            raise RuntimeError(f'Unhandled number of qubits {len(qubits)}')

        matrix = np.moveaxis(np.tensordot(gate, matrix, (in_axes, qubits)),
                             tuple(range(len(qubits))), qubits)

    return matrix


def schedule_to_matrix(
    schedule: Union[Schedule, ScheduleBlock],
    backend: Backend,
    calibrations: Calibrations,
    physical_qubits: Sequence[int],
    qutrits: Sequence[int],
    qutrit_detunings: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Compute the qubit-qutrit-qubit unitary matrix given by the schedule."""
    physical_qubits = tuple(physical_qubits)
    dims = np.full(len(physical_qubits), 2)
    dims[qutrits] = 3
    detuning = np.zeros(len(physical_qubits))
    if qutrit_detunings:
        detuning[qutrits] = qutrit_detunings

    drive_channels = {backend.drive_channel(pq): iq for iq, pq in enumerate(physical_qubits)}
    control_channels = {}
    for iq1, pq1 in enumerate(physical_qubits):
        for iq2, pq2 in enumerate(physical_qubits):
            if iq1 == iq2:
                continue
            try:
                ch = backend.control_channel((pq1, pq2))[0]
            except BackendConfigurationError:
                continue
            control_channels[ch] = (iq1, iq2)

    channel_phases = {ch: 0. for ch in list(drive_channels.keys()) + list(control_channels.keys())}

    starks = {
        op: {
            iq: np.exp(-0.5j * calibrations.get_parameter_value(f'delta_{op}', physical_qubits[iq]))
            for iq in qutrits
        }
        for op in ['x', 'x12', 'sx', 'sx12']
    }
    starks['rzx45p_rotary'] = {}
    for qutrit in qutrits:
        for qubit in [qutrit - 1, qutrit + 1]:
            key = tuple(physical_qubits[iq] for iq in [qubit, qutrit])
            try:
                delta = calibrations.get_parameter_value('delta_rzx45p_rotary', key)
            except CalibrationError:
                continue
            starks['rzx45p_rotary'][(qubit, qutrit)] = np.exp(-0.5j * delta)

    qutrit_freq_shifts = {
        iq: get_qutrit_freq_shift(physical_qubits[iq], backend.target, calibrations)
        for iq in qutrits
    }

    x3 = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]], dtype=complex)
    rx45p3 = scipy.linalg.expm(-0.5j * np.pi / 4. * x3)
    rx45m3 = scipy.linalg.expm(0.5j * np.pi / 4. * x3)
    zero3 = np.zeros((3, 3), dtype=complex)

    def block_matrix(blocks):
        return np.concatenate([np.concatenate(row, axis=1) for row in blocks], axis=0)

    op_unitaries = {
        'x': {
            2: np.array([[0., -1.j], [-1.j, 0.]]),
            3: np.array([[0., -1.j, 0.], [-1.j, 0., 0.], [0., 0., 1]])
        },
        'sx': {
            2: np.array([[1., -1.j], [-1.j, 1.]]) / np.sqrt(2.),
            3: np.array([[1., -1.j, 0.], [-1.j, 1., 0.], [0., 0., np.sqrt(2.)]]) / np.sqrt(2.)
        },
        'x12': {
            3: np.array([[1., 0., 0.], [0., 0., -1.j], [0., -1.j, 0.]])
        },
        'sx12': {
            3: np.array([[np.sqrt(2.), 0., 0.], [0., 1., -1.j], [0., -1.j, 1.]]) / np.sqrt(2.)
        },
        'rzx45p': {
            (2, 3): block_matrix([[rx45p3, zero3], [zero3, rx45m3]])
        },
        'rzx45m': {
            (2, 3): block_matrix([[rx45m3, zero3], [zero3, rx45p3]])
        },
        'rzx45p_rotary': {
            3: np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=complex)
        }
    }

    def phase_shift_op(phase, dim, subspace=0):
        match (dim, subspace):
            case (2, 0):
                return np.array([[np.exp(-1.j * phase), 0.], [0., 1.]])
            case (3, 0):
                return np.array(
                    [[np.exp(-1.j * phase), 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]
                )
            case (3, 1):
                return np.array(
                    [[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., np.exp(1.j * phase)]]
                )

    def embed_1q_gate(gate_name, channel, time):
        iq = drive_channels[channel]
        if dims[iq] == 3:
            bare_op = op_unitaries[gate_name][3].copy()
            phase = channel_phases[channel]
            if gate_name in ['x12', 'sx12']:
                bare_op[0, 0] *= starks[gate_name][iq]
                # x12 pulse at time t corresponds to a pi pulse with phase offset -freq_shift*t
                # in the EF frame
                phase -= (twopi * qutrit_freq_shifts[iq] - detuning[iq]) * time
                subspace = 1
            else:
                bare_op[2, 2] *= starks[gate_name][iq]
                subspace = 0
            rz = phase_shift_op(phase, 3, subspace)
        else:
            bare_op = op_unitaries[gate_name][2].copy()
            rz = phase_shift_op(channel_phases[channel], 2)

        return _embed_1q(iq, bare_op, rz)
    
    def embed_rotary(channel, control):
        iq = drive_channels[channel]
        bare_op = op_unitaries['rzx45p_rotary'][3].copy()
        bare_op[2, 2] *= starks['rzx45p_rotary'][(control, iq)]
        rz = phase_shift_op(channel_phases[channel], 3)
        return _embed_1q(iq, bare_op, rz)

    def _embed_1q(iq, bare_op, rz):
        matrix = np.eye(np.prod(dims[:iq]), dtype=complex)
        matrix = np.kron(matrix, rz.conjugate() @ bare_op @ rz)
        return np.kron(matrix, np.eye(np.prod(dims[iq + 1:]), dtype=complex))

    def embed_2q_gate(gate_name, channel):
        iqc, iqt = control_channels[channel]
        bare_op = op_unitaries[gate_name][(dims[iqc], dims[iqt])].copy()
        block_rz = phase_shift_op(channel_phases[channel], dims[iqt])
        blocks = np.zeros((dims[iqc], dims[iqc], dims[iqt], dims[iqt]), dtype=complex)
        blocks[np.arange(dims[iqc]), np.arange(dims[iqc])] = block_rz
        rz = block_matrix(blocks)
        op = rz.conjugate() @ bare_op @ rz
        if iqc > iqt:
            op = op.reshape((dims[iqc], dims[iqt], dims[iqc], dims[iqt])).transpose((1, 0, 3, 2))
            op = op.reshape((dims[iqc] * dims[iqt], dims[iqc] * dims[iqt]))
            low = iqt
            high = iqc
        else:
            low = iqc
            high = iqt

        matrix = np.eye(np.prod(dims[:low]), dtype=complex)
        matrix = np.kron(matrix, op)
        return np.kron(matrix, np.eye(np.prod(dims[high + 1:]), dtype=complex))

    tmax = 0

    sched_unitary = np.eye(np.prod(dims), dtype=complex)

    for time, inst in schedule.instructions:
        tmax = max(time + inst.duration, tmax)

        if isinstance(inst, (pulse.Delay, pulse.instructions.RelativeBarrier)):
            continue
        if isinstance(inst, pulse.ShiftPhase):
            try:
                logger.debug('Updating phase of %s %f -> %f', inst.channel,
                             channel_phases[inst.channel], channel_phases[inst.channel] + inst.phase)
                channel_phases[inst.channel] += inst.phase
            except KeyError:
                pass
            continue
        if not isinstance(inst, pulse.Play):
            raise RuntimeError(f'Unhandled {inst} at time {time}')

        if isinstance(inst.channel, pulse.DriveChannel):
            if inst.name.startswith('Xp'):
                op_unitary = embed_1q_gate('x', inst.channel, time)
            elif inst.name.startswith('X90p'):
                op_unitary = embed_1q_gate('sx', inst.channel, time)
            elif inst.name.startswith('Ξp'):
                op_unitary = embed_1q_gate('x12', inst.channel, time)
            elif inst.name.startswith('Ξ90p'):
                op_unitary = embed_1q_gate('sx12', inst.channel, time)
            elif inst.name == 'DD':
                op_unitary = embed_1q_gate('x', inst.channel, time)
                op_unitary = op_unitary @ op_unitary
            elif inst.name.startswith('CR90p_d') or inst.name.startswith('CR90m_d'):
                control_channel_name = inst.name.split('_')[2]
                control = control_channels[pulse.ControlChannel(int(control_channel_name[1:]))][0]
                logger.debug('Rotary drive for CR%d->%d on %s', control,
                             drive_channels[inst.channel], control_channel_name)
                op_unitary = embed_rotary(inst.channel, control)
            else:
                raise RuntimeError(f'Unhandled 1q drive {inst} at time {time}')
        elif isinstance(inst.channel, pulse.ControlChannel):
            if inst.name.startswith('CR90p_u'):
                op_unitary = embed_2q_gate('rzx45p', inst.channel)
            elif inst.name.startswith('CR90m_u'):
                op_unitary = embed_2q_gate('rzx45m', inst.channel)
            else:
                raise RuntimeError(f'Unhandled control drive {inst} at time {time}')
        else:
            raise RuntimeError(f'Instruction {inst} on unknown channel')

        logger.debug('%d, %s\n%s', time, inst, op_unitary)
        sched_unitary = op_unitary @ sched_unitary

    rzs = [None] * len(physical_qubits)
    for channel, phase in channel_phases.items():
        if isinstance(channel, pulse.DriveChannel):
            logger.debug('Channel %s final phase %f', channel, phase)
            iq = drive_channels[channel]
            rzs[iq] = phase_shift_op(phase, dims[iq])

    final_rz = np.ones(1, dtype=complex)
    for rz in rzs:
        final_rz = np.kron(final_rz, rz)
    logger.debug('Final Rz\n%s', final_rz)
    sched_unitary = final_rz @ sched_unitary

    sched_unitary.real = np.where(np.abs(sched_unitary.real) < 1.e-8, 0., sched_unitary.real)
    sched_unitary.imag = np.where(np.abs(sched_unitary.imag) < 1.e-8, 0., sched_unitary.imag)

    return sched_unitary
