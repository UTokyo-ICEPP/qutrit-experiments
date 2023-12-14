import cmath
import math
from abc import abstractmethod
from typing import Sequence, List, Union, Optional, Dict

import numpy as np
from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence
from qiskit import QuantumCircuit
from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    Measure
)
from qiskit.providers import Backend
from qiskit.quantum_info import random_clifford, Clifford
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.randomized_benchmarking.rb_analysis import RBAnalysis
from scipy.linalg import det

from ..common.iq_classification import IQClassification
from ..common.ef_space import EFSpaceExperiment
from ..common.gates import X12Gate, SX12Gate, RZ12Gate

class BaseRB1Q(BaseExperiment):
    """Base class for the 1Q qubit RB."""

    @classmethod
    def _default_experiment_options(cls):
        options = super()._default_experiment_options()
        options.lengths = np.arange(1, 100, 7)
        options.num_samples = 5
        options.seed = None
        options.sequences: Dict[int, Sequence[Tuple[float, ...]]] = None

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None,
        num_samples: Optional[int] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
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

    @classmethod
    @abstractmethod
    def generate_sequence(cls, length: int, rng: Generator) -> Sequence[float]:
        """A helper function to return random gate sequence generator."""
        pass

    @classmethod
    @abstractmethod
    def _sequence_to_instructions(cls, *params):
        """A helper function to translate single gate element to basis gate sequence.

        This overrules standard Qiskit transpile protocol and immediately
        apply hard-coded decomposition with respect to the backend basis gates.
        Note that this decomposition ignores global phase.
        """
        pass

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        opt = self.experiment_options

        if opt.sequences is None:
            rng = np.random.default_rng(seed=opt.seed)
            sequences = dict()
            for length in opt.lengths:
                sequences[length] = list(params for params in self.generate_sequence(length, rng))
        else:
            sequences = opt.sequences

        qregs = QuantumRegister(1, name="q")
        cregs = ClassicalRegister(1, name="c")
        qubit = qregs[0]
        clbit = cregs[0]
        exp_circs = []
        for sample_ind in range(opt.num_samples):
            for length in opt.lengths:
                rb_circ = QuantumCircuit(qregs, cregs)
                rb_circ.metadata = {
                    "xval": length,
                    "sample": sample_ind,
                    "qubits": self.physical_qubits
                }

                for params in sequences[length]:
                    for inst in self._sequence_to_instructions(*params):
                        rb_circ._append(inst, [qubit], [])
                    rb_circ.barrier()
                rb_circ._append(Measure(), [qubit], [clbit])
                exp_circs.append(rb_circ)

        return exp_circs


class BaseEFRB1Q(EFSpaceExperiment, IQClassification, BaseRB1Q):
    pass


class EFStandardRB1Q(BaseEFRB1Q):
    """Standard randomized benchmarking for single qutrit Clifford circuit."""
    @classmethod
    def generate_sequence(cls, length: int, rng: Generator):
        """A helper function to return random Clifford sequence generator."""
        composed = Clifford([[1, 0], [0, 1]])
        for _ in range(length):
            elm = random_clifford(1, rng)
            composed = composed.compose(elm)
            yield cls._to_parameters(elm)
        if length > 0:
            yield cls._to_parameters(composed.adjoint())

    @staticmethod
    def _to_parameters(elm: Clifford):
        mat = elm.to_matrix()

        su_mat = det(mat) ** (-0.5) * mat
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda2 = cmath.phase(su_mat[1, 1])
        phimlambda2 = cmath.phase(su_mat[1, 0])
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2

        return theta, phi, lam

    @classmethod
    def _sequence_to_instructions(cls, theta, phi, lam):
        """Single qubit Clifford decomposition with fixed number of physical gates.

        This overrules standard Qiskit transpile protocol and immediately
        apply hard-coded decomposition with respect to the backend basis gates.
        Note that this decomposition ignores global phase.
        """
        return [
            RZ12Gate(lam),
            SX12Gate(),
            RZ12Gate(theta + math.pi),
            SX12Gate(),
            RZ12Gate(phi - math.pi),
        ]


class EFPolarRB1Q(BaseEFRB1Q):
    """Modified randomized benchmarking for ping-pong like random circuit."""
    @classmethod
    def generate_sequence(cls, length: int, rng: Generator):
        """A helper function to return random Clifford sequence generator."""
        for _ in range(length):
            angle1, angle2 = rng.uniform(-np.pi, np.pi, 2)
            yield angle1, angle2

    @classmethod
    def _sequence_to_instructions(cls, angle1, angle2):
        """Single qubit Clifford decomposition with fixed number of physical gates.

        This overrules standard Qiskit transpile protocol and immediately
        apply hard-coded decomposition with respect to the backend basis gates.
        Note that this decomposition ignores global phase.
        """
        return [
            RZ12Gate(angle1),
            X12Gate(),
            RZ12Gate(angle2 - angle1),
            X12Gate(),
            RZ12Gate(-angle2),
        ]
