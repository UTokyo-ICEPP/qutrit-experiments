from typing import List, Optional, Sequence, Tuple
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.circuit import CircuitInstruction, ClassicalRegister
from qiskit.circuit.library import XGate
from qiskit.result import Counts
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, BaseExperiment, Options, ExperimentData
from qiskit_experiments.library import CorrelatedReadoutError as CorrelatedReadoutErrorOrig


from ..common.transpilation import map_to_physical_qubits
from ..common.gates import X12Gate
from ..common.util import default_shots

class CorrelatedReadoutError(CorrelatedReadoutErrorOrig):
    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.shots = 10000
        return options

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        circuits = self.circuits()
        first_circuit = map_to_physical_qubits(circuits[0], self.physical_qubits,
                                               self.transpile_options.target)

        transpiled_circuits = [first_circuit]
        # first_circuit is for 0
        for circuit in circuits[1:]:
            tcirc = first_circuit.copy()
            state_label = circuit.metadata['state_label']
            for i, val in enumerate(reversed(state_label)):
                if val == "1":
                    tcirc.data.insert(0, CircuitInstruction(XGate(), [self.physical_qubits[i]]))

            tcirc.metadata = circuit.metadata
            transpiled_circuits.append(tcirc)

        return transpiled_circuits

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[Counts]:
        shots = self.run_options.get('shots', default_shots)

        template = '{:0%db}' % self.num_qubits

        data = list()
        for state in range(2 ** self.num_qubits):
            data.append(Counts({template.format(state): shots}))

        return data


class MCMLocalReadoutError(BaseExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=MCMLocalReadoutErrorAnalysis(BaseAnalysis), backend=backend)

    def circuits(self) -> List[QuantumCircuit]:
        template = QuantumCircuit(1, 1)
        template.metadata = {
            "experiment_type": self._type,
            "qubit": self.physical_qubits[0],
        }

        circuits = []

        circ = template.copy()
        circ.measure(0, 0)
        circ.metadata['state_label'] = '0'
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.measure(0, 0)
        circ.metadata['state_label'] = '1'
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.append(X12Gate(), [0])
        circ.measure(0, 0)
        circ.metadata['state_label'] = '2'
        circuits.append(circ)

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        circuits = super()._transpiled_circuits()

        for circ in circuits:
            creg = ClassicalRegister(size=2)
            circ.remove_final_measurements(inplace=True)
            circ.add_register(creg)
            # Post-transpilation - logical qubit = physical qubit
            circ.measure(self.physical_qubits[0], creg[0])
            circ.x(self.physical_qubits[0])
            circ.measure(self.physical_qubits[0], creg[1])

        return circuits


def MCMLocalReadoutErrorAnalysis(BaseAnalysis):
#def MCMLocalReadoutErrorAnalysis():
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        assignment_matrix = np.zeros(4, 4)

        for datum in experiment_data.data():
            in_state = int(datum['metadata']['state_label'])
            for obs_state, key in enumerate(['10', '01', '11', '00']):
                assignment_matrix[obs_state, in_state] = datum['counts'].get(key, 0)

            assignment_matrix[:, in_state] /= np.sum(assignment_matrix[:, in_state])

        return [AnalysisResultData(name='assignment_matrix', value=assignment_matrix)], []
