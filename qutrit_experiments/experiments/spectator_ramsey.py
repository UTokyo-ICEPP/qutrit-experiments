"""RamseyXY experiment to measure the frequency shift in a qubit coupled to a qutrit."""
from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Delay, Gate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import Options
from qiskit_experiments.library import RamseyXY

from ..constants import DEFAULT_SHOTS
from ..gates import X12Gate
from ..util.dummy_data import single_qubit_counts

twopi = 2. * np.pi


class SpectatorRamseyXY(RamseyXY):
    """RamseyXY experiment to measure the frequency shift in a qubit coupled to a qutrit."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delay_schedule = None
        options.reverse_qubit_order = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        delay_schedule: Optional[ScheduleBlock] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        if osc_freq is None:
            osc_freq = super()._default_experiment_options().osc_freq

        super().__init__(physical_qubits, backend=backend, delays=delays, osc_freq=osc_freq)
        self.control_state = control_state
        self.set_experiment_options(delay_schedule=delay_schedule)
        if extra_metadata is None:
            self.extra_metadata = {'control_state': control_state}
        else:
            self.extra_metadata = {'control_state': control_state, **extra_metadata}

        self.experiment_index = experiment_index

        self.analysis.set_options(outcome='1')

    def _pre_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        if self.control_state == 1:
            circuit.x(1)
        elif self.control_state == 2:
            circuit.x(1)
            circuit.append(X12Gate(), [1])
        circuit.barrier()
        return circuit

    def _metadata(self):
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)

        return metadata

    def circuits(self) -> list[QuantumCircuit]:
        circuits = super().circuits()

        if self.experiment_options.reverse_qubit_order:
            reversed_circuits = []
            for circuit in circuits:
                reversed_circuit = circuit.copy_empty_like()
                reversed_circuit.compose(circuit, qubits=[1, 0], inplace=True)
                reversed_circuits.append(reversed_circuit)

            circuits = reversed_circuits
            target_qubit = 1
        else:
            target_qubit = 0

        for circuit in circuits:
            circuit.remove_final_measurements()
            creg = ClassicalRegister(1)
            circuit.add_register(creg)
            circuit.measure(target_qubit, creg[0])
            circuit.metadata['readout_qubits'] = [self.physical_qubits[target_qubit]]
            if self.experiment_index is not None:
                circuit.metadata['experiment_index'] = self.experiment_index

        if (delay_sched := self.experiment_options.delay_schedule) is not None:
            delay_param = delay_sched.get_parameters('delay')[0]

            for circuit in circuits:
                delay_inst = next(inst for inst in circuit.data
                                  if isinstance(inst.operation, Delay))
                delay_val = delay_inst.operation.params[0]
                sched = delay_sched.assign_parameters({delay_param: delay_val}, inplace=False)
                delay_inst.operation = Gate('exp(-i*ωz*t)', 1, [delay_val])
                circuit.add_calibration('exp(-i*ωz*t)', [self.physical_qubits[target_qubit]], sched,
                                        [delay_val])

        return circuits

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]: # pylint: disable=unused-argument
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        num_qubits = 1

        tau = 1.e-5
        freq = 2.e+5 + self.experiment_options.osc_freq
        amp = 0.49
        base = 0.51
        phase = 0.1

        delays = np.array(self.experiment_options.delays)

        p_ground_x = 1. - (amp * np.exp(-delays / tau) * np.cos(twopi * freq * delays + phase)
                           + base)
        p_ground_y = 1. - (amp * np.exp(-delays / tau) * np.sin(twopi * freq * delays + phase)
                           + base)
        p_ground = np.empty(2 * delays.shape[0])
        p_ground[::2] = p_ground_x
        p_ground[1::2] = p_ground_y

        return single_qubit_counts(p_ground, shots, num_qubits)
