"""Measurement of phase shift by a diagonal unitary."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.framework import BaseExperiment, Options
import qiskit_experiments.curve_analysis as curve

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins import MapToPhysicalQubits
from ..util.dummy_data import single_qubit_counts

twopi = 2. * np.pi


class PhaseShiftMeasurement(MapToPhysicalQubits, BaseExperiment):
    r"""Phase rotation-Rz-SX experiment.

    Subclasses should implement _phase_rotation_sequence that returns a circuit whose final state
    is of form

    .. math::

        |\psi\rangle = \frac{1}{\sqrt{2}} (|0\rangle - i e^{i\kappa} |1\rangle)
                         \otimes |\text{rest}\rangle.

    The probability of outcome '1' in this experiment is then

    .. math::

        P(1) & = |\langle 1 | \sqrt{X} R_z(\phi) | \psi \rangle|^2 \\
             & = \frac{1}{2} \left( 1 + \cos(\phi + \kappa) \right).

    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.measured_logical_qubit = None
        options.measure_all = False
        options.phase_shifts = np.linspace(0., twopi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        measured_logical_qubit: Optional[int] = None,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=curve.OscillationAnalysis(),
                         backend=backend)

        self.analysis.set_options(
            outcome='1', # Needs to be updated if measure_all option is True
            result_parameters=[curve.ParameterRepr('phase', 'phase_offset')],
            bounds={'amp': (0., 1.)},
            fixed_parameters={'freq': 1. / (2. * np.pi)},
            normalization=False
        )
        self.analysis.plotter.set_figure_options(
            xlabel='Phase shift',
            ylabel='$P(1)$',
            ylim=(-0.1, 1.1)
        )

        if measured_logical_qubit is not None:
            self.set_experiment_options(measured_logical_qubit=measured_logical_qubit)
        if phase_shifts is not None:
            self.set_experiment_options(phase_shifts=phase_shifts)

    def circuits(self) -> list[QuantumCircuit]:
        phi = Parameter('phi')

        nq = len(self._physical_qubits)
        iq = self.experiment_options.measured_logical_qubit or 0
        template = QuantumCircuit(nq, nq if self.experiment_options.measure_all else 1)
        template.compose(self._phase_rotation_sequence(), inplace=True)
        template.barrier()
        template.rz(phi, iq)
        template.sx(iq)
        if self.experiment_options.measure_all:
            template.measure(range(nq), range(nq))
        else:
            template.measure(iq, 0)

        template.metadata = {
            'qubits': self.physical_qubits
        }

        circuits = []
        for phase in self.experiment_options.phase_shifts:
            circuit = template.assign_parameters({phi: phase}, inplace=False)
            circuit.metadata['xval'] = phase
            circuits.append(circuit)

        return circuits

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        return QuantumCircuit(len(self._physical_qubits))

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]: # pylint: disable=unused-argument
        phases = self.experiment_options.phase_shifts + 0.1
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        num_qubits = 1

        one_probs = np.cos(phases) * 0.49 + 0.51

        return single_qubit_counts(one_probs, shots, num_qubits)
