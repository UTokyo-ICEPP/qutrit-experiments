"""Randomized benchmarking of single-qutrit gates."""
from collections.abc import Iterable, Sequence
from abc import abstractmethod
from typing import Any, Optional, Union
import numpy as np
from numpy.random import Generator, RandomState
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitError, Gate
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.providers import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.randomized_benchmarking import (InterleavedRB,
                                                                InterleavedRBAnalysis, StandardRB,
                                                                RBAnalysis)

from ...experiment_mixins import EFSpaceExperiment, MapToPhysicalQubits
from ...gates import RZ12Gate, SX12Gate, X12Gate

SeedType = Union[int, RandomState, Generator]


class UnitaryGate(Gate):
    """A general element of the U(N) group. Not to be confused with Qiskit UnitaryGate."""
    def __init__(self, data: np.ndarray, label: Optional[str] = None):
        assert len(data.shape) == 2 and data.shape[0] == data.shape[1]
        super().__init__('un', 1, [data], label=label)

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be a square-matrix ndarray."""
        if (isinstance(parameter, np.ndarray) and len(parameter.shape) == 2
                and parameter.shape[0] == parameter.shape[1]):
            return parameter
        raise CircuitError(f"invalid parameter {parameter} in gate {self.name}")


def unitary_elements(dim: int, length: int, rng: Generator):
    unitaries = unitary_group.rvs(dim=dim, size=length, random_state=rng)
    if length <= 1:
        unitaries = [unitaries]
    return unitaries


class BaseQutritRB(MapToPhysicalQubits, BaseExperiment):
    """Base class for the 1Q qutrit RB."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.lengths = np.arange(1, 50, 3)
        options.num_samples = 5
        options.seed = None
        options.qubit_mode = None  # for debugging
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
        UnitaryGates, where the latter is decomposed by _append_su().
        """

    def circuits(self) -> list[QuantumCircuit]:
        rng = np.random.default_rng(seed=self.experiment_options.seed)

        circuits = []
        for length in self.experiment_options.lengths:
            for sequence, metadata in self._generate_sequence(length, rng):
                rb_circ = QuantumCircuit(1)
                rb_circ.metadata = {
                    "xval": length,
                    **metadata
                }
                if self.experiment_options.qubit_mode == 1:
                    rb_circ.x(0)
                    rb_circ.barrier()
                for gate in sequence:
                    if isinstance(gate, UnitaryGate):
                        self._append_su(rb_circ, gate, dim_shift=self.experiment_options.qubit_mode)
                    else:
                        rb_circ.append(gate, [0])
                    rb_circ.barrier()
                if self.experiment_options.qubit_mode == 1:
                    rb_circ.x(0)
                rb_circ.measure_all()
                circuits.append(rb_circ)

        return circuits

    @staticmethod
    def _append_su(
        circuit: QuantumCircuit,
        gate: UnitaryGate,
        qubit: int = 0,
        dim_shift: Optional[int] = None
    ):
        r"""Iteratively decompose the SU(2)/SU(3) element into SX, SX12, RZ, and RZ12 gates.

        With :math:`U \in SU(3)` as an example,

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
        :math:`U^{(2)} = G^{(1)}G^{(0)}U^{(0)}`. :math:`U^{(3)}` is a diagonal member of U(3),
        which is given by

        .. math::

            U^{(3)} = R_z(-2(\mathrm{arg}(u_{00}^{(2)})-D))
                      R_{\zeta}(2(\mathrm{arg}(u_{22}^{(2)})-D)).

        with :math:`D = \mathrm{arg}(u_{00}^{(2)}u_{11}^{(2)}u_{22}^{(2)}) / 3`.
        Thus, the decomposition of :math:`U` is

        .. math::

            U = (G^{(0)})^{-1} (G^{(1)})^{-1} (G^{(2)})^{-1}
                R_z(-2\mathrm{arg}(u_{00}^{(3)})) R_{\zeta}(2\mathrm{arg}(u_{22}^{(3)})).

        """
        unitary = gate.params[0]
        dim = unitary.shape[0]
        if dim_shift is None:
            dim_shift = 0

        def embed(subunitary, row):
            embedded = np.eye(dim, dtype=complex)
            embedded[row:row + 2, row:row + 2] = subunitary
            return embedded

        zeroed_elements = [(row, col)
                           for col in range(dim - 1, -1, -1)
                           for row in range(0, col)]

        rot_params = []
        for idx_n in zeroed_elements:
            idx_d = (idx_n[0] + 1, idx_n[1])
            if np.isclose(unitary[idx_d], 0.):
                theta = np.pi
                phi = 0.
            else:
                theta = 2. * np.arctan(np.abs(unitary[idx_n] / unitary[idx_d]))
                phi = -np.angle(unitary[idx_n] / unitary[idx_d])

            cos = np.cos(theta / 2.)
            sin = np.sin(theta / 2.)
            subunitary = np.array(
                [[cos, -np.exp(-1.j * phi) * sin],
                 [np.exp(1.j * phi) * sin, cos]]
            )
            unitary = embed(subunitary, idx_n[0]) @ unitary
            rot_params.append((idx_n[0], theta, phi))

        diag_phases = np.angle(np.diagonal(unitary))
        # global phase correction
        diag_phases -= np.mean(diag_phases)

        rz_gates = [RZGate, RZ12Gate]
        sx_gates = [SXGate, SX12Gate]

        def append_phase_gates(diag_phases):
            phase = 0.
            for idim in range(dim - 1):
                phase += diag_phases[idim]
                circuit.append(rz_gates[idim + dim_shift](-2 * phase), [qubit])

        append_phase_gates(diag_phases)

        for row, theta, phi in rot_params[::-1]:
            # Givens matrices above correspond to U(θ,φ,-φ) in the respective qubit space.
            # To reconstruct the original unitary, we recursively apply
            # U(-θ,φ,-φ) = [U(θ,φ,-φ)]^{-1}.
            # The standard decomposition of U(θ,φ,λ) is
            # U(θ,φ,λ) = P(φ+π) √X Rz(θ+π) √X P(λ)
            # where P(φ) is
            # diag(1, e^{iφ}, 1) ~ exp(diag(-i/3φ, 2i/3φ, -i/3φ))  (g-e space)
            # diag(1, 1, e^{iφ}) ~ exp(diag(-i/3φ, -i/3φ, 2i/3φ))  (e-f space)

            # P(-φ)
            diag_phases = np.full(dim, phi / dim)
            diag_phases[row + 1] -= phi
            append_phase_gates(diag_phases)
            # SX
            circuit.append(sx_gates[row + dim_shift](), [qubit])
            # Rz
            circuit.append(rz_gates[row + dim_shift](-theta + np.pi), [qubit])
            # SX
            circuit.append(sx_gates[row + dim_shift](), [qubit])
            # P(φ+π)
            diag_phases = np.full(dim, -(phi + np.pi) / dim)
            diag_phases[row + 1] += phi + np.pi
            append_phase_gates(diag_phases)


class QutritRB(BaseQutritRB):
    """Standard randomized benchmarking for single qutrit circuit."""
    def _generate_sequence(
        self,
        length: int,
        rng: Generator
    ) -> Iterable[tuple[list[Gate], dict[str, Any]]]:
        """A helper function to return random Clifford sequence generator."""
        dim = 3 if self.experiment_options.qubit_mode is None else 2
        for isample in range(self.experiment_options.num_samples):
            sequence = []
            product = np.eye(dim, dtype=complex)
            for unitary in unitary_elements(dim, length, rng):
                sequence.append(UnitaryGate(unitary))
                product = unitary @ product
            sequence.append(UnitaryGate(np.linalg.inv(product)))
            yield sequence, {'sample': isample}


class QutritInterleavedRB(BaseQutritRB):
    """Interleaved randomized benchmarking for a single qutrit gate."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit_order = 'RIRIRI'
        return options

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
        dim = 3 if self.experiment_options.qubit_mode is None else 2
        if self.experiment_options.qubit_mode == 1:
            gate_unitary = self.gate_unitary[1:3, 1:3]
        else:
            gate_unitary = self.gate_unitary[:dim, :dim]

        def reference():
            sequence = []
            product = np.eye(dim, dtype=complex)
            for unitary in unitary_elements(dim, length, rng):
                sequence.append(UnitaryGate(unitary))
                product = unitary @ product
            sequence.append(UnitaryGate(np.linalg.inv(product)))
            return sequence

        def interleaved():
            sequence = []
            product = np.eye(dim, dtype=complex)
            for unitary in unitary_elements(dim, length, rng):
                sequence.append(UnitaryGate(unitary))
                sequence.append(self.interleaved_gate())
                product = gate_unitary @ unitary @ product
            sequence.append(UnitaryGate(np.linalg.inv(product)))
            return sequence

        def double_interleaved():
            sequence = []
            product = np.eye(dim, dtype=complex)
            for unitary in unitary_elements(dim, length, rng):
                sequence.append(UnitaryGate(unitary))
                sequence.append(self.interleaved_gate())
                sequence.append(self.interleaved_gate())
                product = gate_unitary @ gate_unitary @ unitary @ product
            sequence.append(UnitaryGate(np.linalg.inv(product)))
            return sequence

        if self.experiment_options.circuit_order == 'RIRIRI':
            for isample in range(self.experiment_options.num_samples):
                yield reference(), {'sample': isample, 'interleaved': False}
                yield interleaved(), {'sample': isample, 'interleaved': True}
        elif self.experiment_options.circuit_order == 'RRRIII':
            for isample in range(self.experiment_options.num_samples):
                yield reference(), {'sample': isample, 'interleaved': False}
            for isample in range(self.experiment_options.num_samples):
                yield interleaved(), {'sample': isample, 'interleaved': True}
        elif self.experiment_options.circuit_order == 'RDRDRD':
            for isample in range(self.experiment_options.num_samples):
                yield reference(), {'sample': isample, 'interleaved': False}
                yield double_interleaved(), {'sample': isample, 'interleaved': True}


GATE_UNITARIES = {
    XGate: np.array(
        [[0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.]]
    ),
    SXGate: np.array(
        [[(1. + 1.j) / 2., (1. - 1.j) / 2., 0.],
         [(1. - 1.j) / 2., (1. + 1.j) / 2., 0.],
         [0., 0., 1.]]
    ),
    X12Gate: np.array(
        [[1., 0., 0.],
         [0., 0., 1.],
         [0., 1., 0.]]
    ),
    SX12Gate: np.array(
        [[1., 0., 0.],
         [0., (1. + 1.j) / 2., (1. - 1.j) / 2.],
         [0., (1. - 1.j) / 2., (1. + 1.j) / 2.]]
    )
}


class EFStandardRB(EFSpaceExperiment, StandardRB):
    """StandardRB in the EF space."""
    _decompose_before_cast = True


class EFInterleavedRB(EFSpaceExperiment, InterleavedRB):
    """InterleavedRB in the EF space."""
    _decompose_before_cast = True
