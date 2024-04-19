from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ..experiment_mixins import MapToPhysicalQubits


class PhaseTable(MapToPhysicalQubits, BaseExperiment):
    """Measure the phase of diagonals of a unitary."""
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
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, backend=backend)
        self.set_experiment_options(circuit=circuit)
        self.analysis = None

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []
        num_qubits = len(self._physical_qubits)
        phi = Parameter('phi')
        for qubit in range(num_qubits):
            for init in range(2 ** (num_qubits - 1)):
                bits = [(init >> iq) % 2 for iq in range(num_qubits - 1)]
                bits.insert(qubit, 0)
                template = QuantumCircuit(num_qubits)
                for iq, bit in enumerate(bits):
                    if bit == 1:
                        template.x(iq)
                template.sx(qubit)
                template.barrier()
                template.compose(self.experiment_options.circuit, inplace=True)
                template.barrier()
                template.rz(phi, qubit)
                template.sx(qubit)
                template.measure_all()
                template.metadata = {
                    'measured_qubit': qubit,
                    'init': init,
                }

                for angle in self.experiment_options.angles:
                    circuits.append(template.assign_parameters({phi: angle}, inplace=False))

        return circuits

