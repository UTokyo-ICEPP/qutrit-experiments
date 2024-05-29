"""RamseyXY experiment to measure the frequency shift in a qubit coupled to a qutrit."""
from collections.abc import Sequence
from typing import Any, Optional
from matplotlib.figure import Figure
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Delay, Gate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.library.characterization import RamseyXY, RamseyXYAnalysis

from ...constants import DEFAULT_SHOTS
from ...gates import X12Gate

twopi = 2. * np.pi


class SpectatorRamseyXY(RamseyXY):
    """RamseyXY experiment to measure the frequency shift in a qubit coupled to a qutrit.

    In this experiment, qubit ordering is fixed to control=0, target=1, whereas the measured qubit
    is 0 in RamseyXY. Qubits are reordered post-construction in circuits(). Method _pre_circuit()
    has to be written with control=1, target=0.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.control_state = None
        options.delay_schedule = None
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
        self.analysis = RamseyXYAnalysisOffset()
        self.set_experiment_options(control_state=control_state, delay_schedule=delay_schedule)
        self.extra_metadata = extra_metadata
        self.experiment_index = experiment_index
        self.analysis.set_options(
            outcome='1', # default outcome will be set to '11' without this line
            fixed_parameters={'tau': np.inf}
        )

    def _pre_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        # Control qubit is 1 in _pre_circuit
        if self.experiment_options.control_state == 1:
            circuit.x(1)
        elif self.experiment_options.control_state == 2:
            circuit.x(1)
            circuit.append(X12Gate(), [1])
        circuit.barrier()
        return circuit

    def _metadata(self):
        metadata = super()._metadata()
        metadata['control_state'] = self.experiment_options.control_state
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []
        for original in super().circuits():
            circuit = original.copy_empty_like()
            circuit.compose(original, qubits=[1, 0], inplace=True)
            circuit.remove_final_measurements()
            circuit.barrier()
            if self.experiment_options.control_state == 2:
                circuit.append(X12Gate(), [0])
            creg = ClassicalRegister(1)
            circuit.add_register(creg)
            circuit.measure(1, creg[0])
            if self.experiment_index is not None:
                circuit.metadata['experiment_index'] = self.experiment_index
            circuits.append(circuit)

        if (delay_sched := self.experiment_options.delay_schedule) is not None:
            delay_param = delay_sched.get_parameters('delay')[0]

            for circuit in circuits:
                delay_inst = next(inst for inst in circuit.data
                                  if isinstance(inst.operation, Delay))
                delay_val = delay_inst.operation.params[0]
                sched = delay_sched.assign_parameters({delay_param: delay_val}, inplace=False)
                delay_inst.operation = Gate('exp(-iωzt)', 2, [delay_val])
                delay_inst.qubits = tuple(circuit.qregs[0])
                circuit.add_calibration('exp(-iωzt)', self.physical_qubits, sched, [delay_val])

        return circuits


class RamseyXYAnalysisOffset(RamseyXYAnalysis):
    """RamseyXYAnalysis with additional analysis result (osc_freq subtracted freq)."""
    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_analysis(experiment_data)
        freq = next(res.value for res in analysis_results if res.name == 'freq')
        freq_offset = experiment_data.data(0)['metadata']['osc_freq']
        analysis_results.append(
            AnalysisResultData(name='ramsey_freq', value=freq - freq_offset)
        )
        return analysis_results, figures
