"""Randomized benchmarking of single-qutrit gates."""
from collections.abc import Iterable, Sequence
from abc import abstractmethod
from typing import Any, Optional, Union
import numpy as np
from numpy.random import Generator, RandomState
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import SXGate, XGate
from qiskit.providers import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.randomized_benchmarking import InterleavedRBAnalysis, RBAnalysis
from scipy.linalg import det
from scipy.stats import unitary_group

from ..experiment_mixins import MapToPhysicalQubits
from ..gates import RZ12Gate, SX12Gate, X12Gate

SeedType = Union[int, RandomState, Generator]


class SU3Gate(Gate):
    """A general element of the SU(3) group."""
    def __init__(self, data: np.ndarray, label: Optional[str] = None):
        assert data.shape == (3, 3)
        super().__init__('su3', 1, [data], label=label)

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")


def su3_elements(length: int, rng: Generator):
    unitaries = unitary_group.rvs(dim=3, size=length, random_state=rng)
    if length <= 1:
        unitaries = [unitaries]
    # Renormalize by the cubic root of the determinant
    determinants = np.array([det(unitary) for unitary in unitaries])
    unitaries *= np.exp(-1.j * np.angle(determinants) / 3.)[:, None, None]
    return unitaries


class BaseQutritRB(MapToPhysicalQubits, BaseExperiment):
    """Base class for the 1Q qutrit RB."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.lengths = np.arange(1, 100, 7)
        options.num_samples = 5
        options.seed = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None,
        num_samples: Optional[int] = None,
        seed: Optional[SeedType] = None,
    ):
        """Initialize an experiment.

        Args:
            qubit: A physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
        """
        analysis = RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio=None)
        analysis.plotter.set_figure_options(
            xlabel="Sequence Length",
            ylabel="Survival Probability",
        )
        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        options = {}
        if lengths is not None:
            options["lengths"] = lengths
        if num_samples is not None:
            options["num_samples"] = num_samples
        if seed is not None:
            options["seed"] = seed
        if options:
            self.set_experiment_options(**options)

    @abstractmethod
    def _generate_sequence(
        self,
        length: int,
        rng: Generator
    ) -> Iterable[tuple[list[Gate], dict[str, Any]]]:
        """Generate the sequence of gate units.

        Subclasses must implement this function. Each outermost entry defines a circuit by the gate
        sequence and a metadata dict. The gate sequence must consist of backend basis gates and
        SU3Gates, where the latter is decomposed by _append_su3().
        """

    def circuits(self) -> list[QuantumCircuit]:
        rng = np.random.default_rng(seed=self.experiment_options.seed)

        circuits = []
        for length in self.experiment_options.lengths:
            for sequence, metadata in self._generate_sequence(length, rng):
                rb_circ = QuantumCircuit(1)
                rb_circ.metadata = {
                    "xval": length,
                    "qubits": self.physical_qubits
                }
                rb_circ.metadata.update(metadata)
                for gate in sequence:
                    if isinstance(gate, SU3Gate):
                        self._append_su3(rb_circ, gate)
                    else:
                        rb_circ.append(gate, [0])
                rb_circ.measure_all()
                circuits.append(rb_circ)

        return circuits

    @staticmethod
    def _append_su3(circuit: QuantumCircuit, gate: SU3Gate, qubit: int = 0):
        r"""Iteratively decompose the SU(3) element into SX, SX12, RZ, and RZ12 gates.

        Given a unitary

        .. math::

            U = U^{(0)} = \begin{pmatrix}
                            u_{00}^{(0)} & u_{01}^{(0)} & u_{02}^{(0)} \\
                            u_{10}^{(0)} & u_{11}^{(0)} & u_{12}^{(0)} \\
                            u_{20}^{(0)} & u_{21}^{(0)} & u_{22}^{(0)}
                          \end{pmatrix}

        we first find a Givens rotation

        .. math::

            G^{(0)} & = \begin{pmatrix}
                          \cos(\theta^{(0)}/2) & -e^{-i\phi^{(0)}} \sin(\theta^{(0)}/2) & 0 \\
                          e^{i\phi^{(0)}} \sin(\theta^{(0)}/2) & \cos(\theta^{(0)}/2)   & 0 \\
                          0                              & 0                            & 1
                        \end{pmatrix} \\
                    & = R_z(\phi - \pi/2) SX R_z(\pi - \theta) SX R_z(-\phi - \pi/2)

        that eliminates the upper right element:

        .. math::

            G^{(0)} (u_{02}^{(0)}, u_{12}^{(0)}, u_{22}^{(0)})^T
            = (0, u_{12}^{(1)}, u_{22}^{(1)})^T.

        The parameters of :math:`G^{(0)}` are uniquely determined by

        .. math::

            & \cos(\theta^{(0)}/2) u_{02}^{(0)} - e^{-i\phi^{(0)}} \sin(\theta^{(0)}/2) u_{12}^{(0)}
            = 0 \\
            \Rightarrow
            & \theta^{(0)}
            = 2\mathrm{arctan}\left(\left|\frac{u_{02}^{(0)}}{u_{12}^{(0)}}\right|\right), \\
            & \phi^{(0)}
            = -\mathrm{arg}\left(\frac{u_{02}^{(0)}}{u_{12}^{(0)}}\right)

        if :math:`u_{12}^{(0)} \neq 0`. Otherwise :math:`\theta^{(0)} = \pi` and :math:`\phi^{(0)}`
        is undetermined.

        Similarly, rotation :math:`G^{(1)}` eliminates the 1,2 element of
        :math:`U^{(1)} = G^{(0)}U^{(0)}`, and :math:`G^{(2)}` the 0,1 element of
        :math:`U^{(2)} = G^{(1)}G^{(0)}U^{(0)}`. :math:`U^{(3)}` is a diagonal element of SU(3),
        which is given by

        .. math::

            U^{(3)} = R_z(-2\mathrm{arg}(u_{00}^{(2)})) R_{\zeta}(2\mathrm{arg}(u_{22}^{(2)})).

        Thus, the decomposition of :math:`U` is

        .. math::

            U = (G^{(0)})^{-1} (G^{(1)})^{-1} (G^{(2)})^{-1}
                R_z(-2\mathrm{arg}(u_{00}^{(3)})) R_{\zeta}(2\mathrm{arg}(u_{22}^{(3)})).

        """
        unitary = gate.params[0]
        rot_params = []
        for idx_n in [(0, 2), (1, 2), (0, 1)]:
            idx_d = (idx_n[0] + 1, idx_n[1])
            if np.isclose(unitary[idx_d], 0.):
                theta = np.pi
                phi = 0.
            else:
                theta = 2. * np.arctan(np.abs(unitary[idx_n] / unitary[idx_d]))
                phi = -np.angle(unitary[idx_n] / unitary[idx_d])

            cos = np.cos(theta / 2.)
            sin = np.sin(theta / 2.)
            phase = np.exp(1.j * phi)
            if idx_n[0] == 0:
                givens = np.array(
                    [[cos, -np.conjugate(phase) * sin, 0.],
                     [phase * sin, cos, 0.],
                     [0., 0., 1.]]
                )
            else:
                givens = np.array(
                    [[1., 0., 0.],
                     [0., cos, -np.conjugate(phase) * sin],
                     [0., phase * sin, cos]]

                )

            unitary = givens @ unitary
            rot_params.append((idx_n[0], theta, phi))

        diag_phases = np.angle(np.diagonal(unitary))
        circuit.rz(-2 * diag_phases[0], qubit)
        circuit.append(RZ12Gate(2. * diag_phases[2]), [qubit])
        for space, theta, phi in rot_params[::-1]:
            # Givens matrices above correspond to U(θ,φ,-φ) in the respective qubit space.
            # To reconstruct the original unitary, we recursively apply
            # U(-θ,φ,-φ) = [U(θ,φ,-φ)]^{-1}.
            # The standard decomposition of U(θ,φ,λ) is
            # U(θ,φ,λ) = P(φ+π) √X Rz(θ+π) √X P(λ)
            # where
            # √X = 1/2 [[1+i, 1-i], [1-i, 1+i]] = e^{iπ/4} Rx(π/2).
            # The qubit-space global phase e^{iπ/4} is a geometric phase in the qutrit space. In
            # fact our SX gate is implemented more like Rx(π/2), so we express U in terms of Rz and
            # Rx:
            # U(θ,φ,λ) = -e^{i(φ+λ)/2} Rz(φ+π) Rx(π/2) Rz(θ+π) Rx(π/2) Rz(λ).
            # So for U(-θ,φ,-φ) we have
            # U(-θ,φ,-φ) = -Rz(φ+π) Rx(π/2) Rz(-θ+π) Rx(π/2) Rz(-φ)
            # i.e. the qubit-space gate sequence results in -U(-θ,φ,-φ).
            if space == 0:
                circuit.rz(-phi, qubit)
                circuit.sx(qubit)
                circuit.rz(-theta + np.pi, qubit)
                circuit.sx(qubit)
                circuit.rz(phi + np.pi, qubit)
                # geometric phase correction
                circuit.rz(2. * np.pi / 3., qubit)
                circuit.append(RZ12Gate(4. * np.pi / 3.), [qubit])
            else:
                circuit.append(RZ12Gate(-phi), [qubit])
                circuit.append(SX12Gate(), [qubit])
                circuit.append(RZ12Gate(-theta + np.pi), [qubit])
                circuit.append(SX12Gate(), [qubit])
                circuit.append(RZ12Gate(phi + np.pi), [qubit])
                # geometric phase correction
                circuit.rz(-4. * np.pi / 3., qubit)
                circuit.append(RZ12Gate(-2. * np.pi / 3.), [qubit])


class QutritRB(BaseQutritRB):
    """Standard randomized benchmarking for single qutrit circuit."""
    def _generate_sequence(
        self,
        length: int,
        rng: Generator
    ) -> Iterable[tuple[list[Gate], dict[str, Any]]]:
        """A helper function to return random Clifford sequence generator."""
        for isample in range(self.experiment_options.num_samples):
            sequence = []
            final_unitary = np.eye(3, dtype=complex)
            for unitary in su3_elements(length, rng):
                sequence.append(SU3Gate(unitary))
                final_unitary = unitary @ final_unitary
            sequence.append(SU3Gate(np.linalg.inv(final_unitary)))
            yield sequence, {'sample': isample}


class QutritInterleavedRB(QutritRB):
    """Interleaved randomized benchmarking for a single qutrit gate."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        interleaved_gate: type[Gate],
        gate_unitary: Optional[np.ndarray] = None,
        lengths: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None,
        num_samples: Optional[int] = None,
        seed: Optional[SeedType] = None,
    ):
        super().__init__(physical_qubits, lengths=lengths, backend=backend,
                         num_samples=num_samples, seed=seed)
        self.interleaved_gate = interleaved_gate
        if gate_unitary is None:
            try:
                gate_unitary = GATE_UNITARIES[interleaved_gate]
            except KeyError as ex:
                raise RuntimeError(f'Gate unitary unknown for {interleaved_gate}') from ex
        self.gate_unitary = np.array(gate_unitary)

        self.analysis = InterleavedRBAnalysis()
        self.analysis.set_options(outcome="0")
        self.analysis.plotter.set_figure_options(
            xlabel="Sequence Length",
            ylabel="Survival Probability",
        )

    def _generate_sequence(
        self,
        length: int,
        rng: Generator
    ) -> Iterable[tuple[list[Gate], dict[str, Any]]]:
        """A helper function to return random Clifford sequence generator."""
        for isample in range(self.experiment_options.num_samples):
            sequence = []
            final_unitary = np.eye(3, dtype=complex)
            for unitary in su3_elements(length, rng):
                sequence.append(SU3Gate(unitary))
                final_unitary = unitary @ final_unitary
            sequence.append(SU3Gate(np.linalg.inv(final_unitary)))
            yield sequence, {'sample': isample, 'interleaved': False}

            sequence = []
            final_unitary = np.eye(3, dtype=complex)
            for unitary in su3_elements(length, rng):
                sequence.append(SU3Gate(unitary))
                sequence.append(self.interleaved_gate())
                final_unitary = self.gate_unitary @ unitary @ final_unitary
            sequence.append(SU3Gate(np.linalg.inv(final_unitary)))
            yield sequence, {'sample': isample, 'interleaved': True}


GATE_UNITARIES = {
    XGate: np.array(
        [[0., -1.j, 0.],
         [-1.j, 0., 0.],
         [0., 0., 1.]]
    ),
    SXGate: np.array(
        [[np.sqrt(0.5), -1.j * np.sqrt(0.5), 0.],
         [-1.j * np.sqrt(0.5), np.sqrt(0.5), 0.],
         [0., 0., 1.]]
    ),
    X12Gate: np.array(
        [[1., 0., 0.],
         [0., 0., -1.j],
         [0., -1.j, 0.]]
    ),
    SX12Gate: np.array(
        [[1., 0., 0.],
         [0., np.sqrt(0.5), -1.j * np.sqrt(0.5)],
         [0., -1.j * np.sqrt(0.5), np.sqrt(0.5)]]
    )
}
