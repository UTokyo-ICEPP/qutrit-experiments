from typing import Union, Optional, Iterable, List, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

from ...common.framework_overrides import BatchExperiment, CompoundAnalysis
from ..process_tomography import CircuitTomography


class CCCXNoCRTomography(BatchExperiment):
    """Parallel single-qubit QPT on a four-qubit schedule."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        backend: Optional[Backend] = None
    ):
        x_schedules = [backend.target['x'][(qubit,)].calibration
                       for qubit in physical_qubits]

        experiments = []
        analyses = []

        for iq, qubit in enumerate(physical_qubits):
            phys_qubits = [qubit]
            logical_neighbors = list(range(4))
            logical_neighbors.remove(iq)
            for states in zip(*np.unravel_index(np.arange(8), (2, 2, 2))):
                with pulse.build(name='sched', backend=backend, default_alignment='left') as sched:
                    for iqn, state in zip(logical_neighbors, states):
                        if state == 1:
                            pulse.call(x_schedules[iqn])
                    pulse.barrier(*physical_qubits)
                    pulse.call(schedule)

                circuit = QuantumCircuit(1)
                circuit.append(Gate('sched', 1, []), [0])
                circuit.add_calibration('sched', phys_qubits, sched)
                metadata = {'logical_qubit': iq, 'spectator_states': list(states)}
                exp = CircuitTomography(circuit, QuantumCircuit(1), backend=backend,
                                        physical_qubits=phys_qubits, extra_metadata=metadata)
                experiments.append(exp)
                analyses.append(exp.analysis)

        super().__init__(experiments, analysis=CCCXNoCRTomographyAnalysis(analyses))


class CCCXNoCRTomographyAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List['mpl.figure.Figure']
    ) -> Tuple[List[AnalysisResultData], List['mpl.figure.Figure']]:
        spectator_states = list(zip(*np.unravel_index(np.arange(8), (2, 2, 2))))
        labels = [
            [fr'$|x{q1}{q2}{q3}\rangle$' for q1, q2, q3 in spectator_states],
            [fr'$|{q0}x{q2}{q3}\rangle$' for q0, q2, q3 in spectator_states],
            [fr'$|{q0}{q1}x{q3}\rangle$' for q0, q1, q3 in spectator_states],
            [fr'$|{q0}{q1}{q2}x\rangle$' for q0, q1, q2 in spectator_states]
        ]

        fidelities = []

        component_index = experiment_data.metadata["component_child_index"]
        for idx in component_index:
            child_data = experiment_data.child_data(idx)
            fidelities.append(child_data.analysis_results('process_fidelity').value)

        fig, axs = plt.subplots(1, 4, sharey=True, figsize=[12.8, 4.8])
        for iax, ax in enumerate(axs):
            ax.scatter(labels[iax], fidelities[8 * iax:8 * (iax + 1)])
            ax.tick_params('x', labelrotation=90.)

        return [], [fig]
