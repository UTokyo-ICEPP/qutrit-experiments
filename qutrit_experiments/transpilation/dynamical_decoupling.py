"""Functions and classes for dynamical decoupling."""
from collections.abc import Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import InstructionDurations, TransformationPass


class DDCalculator:
    """Class to calculate delays to be inserted for qubit dynamical decoupling."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        instruction_durations: InstructionDurations,
        pulse_alignment: int
    ):
        self._x_durations = {lq: instruction_durations.get('x', [pq])
                             for lq, pq in enumerate(physical_qubits)}
        self._alignment = pulse_alignment

    def _constrained_length(self, values: Sequence[float]) -> np.ndarray:
        return self._alignment * np.floor(np.asarray(values) / self._alignment).astype(int)

    def calculate_delays(
        self,
        qubit: int,
        duration: int,
        num_pairs: int = 1,
        distribution: str = 'symmetric'
    ) -> list:
        """Compute the delay durations to perform DD with the specified number of X pairs.

        Logic copied from qiskit.transpiler.passes.scheduling.padding.dynamical_decoupling.

        Args:
            qubit: Qubit to apply DD on.
            duration: Overall duration of the idle time.
            num_pairs: Number of X gate pairs to insert.
            distribution: 'left', 'right', or 'symmetric'.
        """
        assert duration % self._alignment == 0, \
            f'Duration {duration} is not a multiple of pulse alignment {self._alignment}'

        num_xs = 2 * num_pairs
        slack = duration - num_xs * self._x_durations[qubit]
        if slack <= 0:
            return

        if distribution == 'left':
            spacing = np.concatenate([[0.], np.full(num_xs, 1. / num_xs)])
            idx_seq = np.arange(num_xs, -1, -1)
        elif distribution == 'right':
            spacing = np.concatenate([np.full(num_xs, 1. / num_xs), [0.]])
            idx_seq = np.arange(num_xs)
        elif distribution == 'symmetric':
            spacing = np.concatenate([[0.5 / num_xs],
                                      np.full(num_xs - 1, 1. / num_xs),
                                      [0.5 / num_xs]])
            idx_seq = np.empty(num_xs + 1, dtype=int)
            idx_seq[::2] = np.arange(num_pairs + 1)
            idx_seq[1::2] = np.arange(num_xs, num_pairs, -1)

        taus = self._constrained_length(slack * spacing)
        extra_slack = slack - np.sum(taus)

        while extra_slack > 0:
            for idx in idx_seq:
                taus[idx] += self._alignment
                extra_slack -= self._alignment
                if extra_slack == 0:
                    break

        return taus.tolist()

    def append_dd(
        self,
        circuit: QuantumCircuit,
        qubit: int,
        duration: int,
        num_pairs: int = 1,
        distribution: str = 'symmetric'
    ) -> None:
        """Calculate the DD delays and append X gates and delays to the QuantumCircuit.

        Args:
            circuit: QuantumCircuit to append the DD sequence to.
            qubit: Qubit to apply DD on.
            duration: Overall duration of the idle time.
            num_pairs: Number of X gate pairs to insert.
            distribution: 'left', 'right', or 'symmetric'.
        """
        taus = self.calculate_delays(qubit, duration, num_pairs, distribution=distribution)
        for tau in taus[:-1]:
            if tau:
                circuit.delay(tau, qubit)
            circuit.x(qubit)
        if taus[-1]:
            circuit.delay(taus[-1], qubit)


class AddDDCalibration(TransformationPass):
    """Add DoubleDrag "pulse" schedules as calibrations of dd gates."""
    def __init__(self):
        super().__init__()
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for placement in ['left', 'right']:
            name = f'dd_{placement}'
            for node in dag.named_nodes(name):
                qubit = dag.find_bit(node.qargs[0]).index
                duration = node.op.params[0]
                if dag.calibrations.get(name, {}).get(((qubit,), (duration,))) is None:
                    sched = self.calibrations.get_schedule(name, qubit,
                                                           assign_params={'duration': duration})
                    dag.add_calibration(name, [qubit], sched, params=[duration])

        return dag
