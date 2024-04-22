"""Measurement of phase entries of a diagonal unitary."""
from collections.abc import Sequence
from typing import Any, Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit_experiments.curve_analysis.standard_analysis import OscillationAnalysis
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ..experiment_mixins import MapToPhysicalQubits
from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment


class DiagonalCircuitPhaseShift(MapToPhysicalQubits, BaseExperiment):
    """Measurement of the phase shift imparted by a diagonal circuit.
    
    In this experiment we fix the states of the spectator qubits to a specific computational basis
    and perform an SX-U-Rz-SX type sequence on the measured qubit. Assuming that U is diagonal and
    therefore acts as pure phases φ0 and φ1 to |0> and |1> of the measured qubit, respectively, the
    probability of observing |1> after the sequence is

    |<1|SX Rz(φ) U SX|0>|^2 = |1/2 [-i<0| + <1|] [exp(i (φ0-φ/2))|0> -i exp(i (φ1+φ/2))|1>]|^2
                            = cos^2(φ0 - φ1 - φ)/2
                            = 1/2 [1 + cos(φ + (φ1 - φ0))]
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit = None
        options.angles = np.linspace(0., 2. * np.pi, 12, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        measured_logical_qubit: int,
        state: tuple[bool, ...],
        backend: Optional[Backend] = None
    ):
        """
        state: Whether the bit should be raised, from low to high excluding the measured bit.
        """
        super().__init__(physical_qubits, backend=backend, analysis=OscillationAnalysis())
        self.measured_logical_qubit = measured_logical_qubit
        self.state = state
        self.set_experiment_options(circuit=circuit)

        outcome_rev = list('1' if xup else '0' for xup in state)
        outcome_rev.insert(measured_logical_qubit, '1')
        
        self.analysis.set_options(
            outcome=''.join(reversed(outcome_rev)),
            result_parameters=['phase'],
            fixed_parameters={'freq': 1.},
            p0={'amp': 0.5, 'base': 0.5},
            bounds={'amp': (0., 1.), 'base': (0., 1.)}
        )

    def circuits(self) -> list[QuantumCircuit]:
        num_qubits = len(self._physical_qubits)
        logical_qubits = list(range(num_qubits))
        logical_qubits.remove(self.measured_logical_qubit)
        phi = Parameter('phi')
        template = QuantumCircuit(num_qubits)
        for bit, xup in zip(logical_qubits, self.state):
            if xup:
                template.x(bit)
        template.sx(self.measured_logical_qubit)
        template.barrier()
        template.compose(self.experiment_options.circuit, inplace=True)
        template.barrier()
        template.rz(phi, self.measured_logical_qubit)
        template.sx(self.measured_logical_qubit)
        template.measure_all()

        circuits = []
        for angle in self.experiment_options.angles:
            circuit = template.assign_parameters({phi: angle}, inplace=False)
            circuit.metadata = {'xval': angle}
            circuits.append(circuit)

        return circuits
    
    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['measured_logical_qubit'] = self.measured_logical_qubit
        metadata['state'] = list(self.state)


class PhaseTable(BatchExperiment):
    """Measure the phase of diagonals of a unitary.
    
    Each of the N*(2**(N-1)) phase shift measurement results corresponds to a phase difference of 
    specific diagonal entries of U with some redundancy. Perform a least-squares fit to obtain the
    2**N - 1 diagonals (entry 0 is fixed to 0).
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None
    ):
        experiments = []

        num_qubits = len(physical_qubits)
        for qubit in range(num_qubits):
            for init in range(2 ** (num_qubits - 1)):
                state = tuple((init >> iq) % 2 == 1 for iq in range(num_qubits - 1))
                experiments.append(
                    DiagonalCircuitPhaseShift(physical_qubits, circuit, qubit, state,
                                              backend=backend)
                )
                
        super().__init__(experiments, backend=backend,
                         analysis=PhaseTableAnalysis([exp.analysis for exp in experiments]))


class PhaseTableAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        idx_high = []
        idx_low = []
        phase_diffs = []
        errs = []
        for child_data in experiment_data.child_data():
            logical_qubit = child_data.metadata['measured_logical_qubit']
            state = child_data.metadata['state']
            num_qubits = len(state) + 1
            idx_low.append(
                sum((int(bit) << iq) for iq, bit in enumerate(state[:logical_qubit]))
                + sum((int(bit) << (iq + logical_qubit + 1))
                      for iq, bit in enumerate(state[logical_qubit:]))
            )
            idx_high.append(idx_low[-1] + (1 << logical_qubit))
            fit_res = child_data.analysis_results('phase').value
            phase_diffs.append(fit_res.n)
            errs.append(fit_res.std_dev)
        phase_diffs = np.array(phase_diffs)
        errs = np.array(errs)

        def fun(diagonals):
            phases = np.concatenate([[0], diagonals])
            return ((phases[idx_high] - phases[idx_low]) - phase_diffs) / errs
        
        result = least_squares(fun, np.zeros(2 ** (num_qubits - 1)))
        diagonals = np.concatenate([[0.], result.x])

        analysis_results.append(AnalysisResultData(name='phases', value=diagonals))

        if self.options.plot:
            num_states = 2 ** num_qubits
            ax = get_non_gui_ax()
            ax.scatter(np.arange(num_states), diagonals)
            ax.set_xlabel('State')
            ax.set_ylabel('Phase')
            ax.set_xticks(np.arange(num_states), labels=[f'{i}' for i in range(num_states)])
            ax.set_ylim(0., 2. * np.pi)
            figures.append(ax.get_figure())

        return analysis_results, figures
