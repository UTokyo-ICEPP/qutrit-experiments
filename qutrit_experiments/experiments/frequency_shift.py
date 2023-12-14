from typing import Any, Dict, List, Sequence, Optional
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit_experiments.framework import Options
from qiskit_experiments.library import RamseyXY
from qiskit_experiments.library.characterization import RamseyXYAnalysis

from ..common.framework_overrides import BatchExperiment
from ..common.linked_curve_analysis import LinkedCurveAnalysis


class RamseyXYFrequencyShift(RamseyXY):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.frequency_shift = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency_shift: float,
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        osc_freq: float = 2e6,
        experiment_index: Optional[int] = None
    ):
        super().__init__(physical_qubits, backend=backend, delays=delays, osc_freq=osc_freq)
        self.set_experiment_options(frequency_shift=frequency_shift)
        self.experiment_index = experiment_index

    def _pre_circuit(self) -> QuantumCircuit:
        with pulse.build(name='shiftfreq') as shiftfreq:
            pulse.shift_frequency(self.experiment_options.frequency_shift,
                                  pulse.DriveChannel(self.physical_qubits[0]))

        circuit = QuantumCircuit(1)
        circuit.append(Gate('shiftfreq', 1, []), [0])
        circuit.add_calibration('shiftfreq', [self.physical_qubits[0]], shiftfreq)
        return circuit

    def circuits(self) -> List[QuantumCircuit]:
        circuits = super().circuits()
        if self.experiment_index is None:
            return circuits

        for circuit in circuits:
            circuit.metadata.update(
                experiment_index=self.experiment_index
            )

        return circuits

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata['frequency_shift'] = self.experiment_options.frequency_shift
        return metadata


class FrequencyShiftClosure(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency_shifts: Sequence[float],
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        osc_freq: float = 2.e+6
    ):
        experiments = []
        analyses = []

        for idx, shift in enumerate(frequency_shifts):
            exp = RamseyXYFrequencyShift(physical_qubits, shift, backend=backend, delays=delays,
                                         osc_freq=osc_freq, experiment_index=idx)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=FrequencyShiftClosureAnalysis(analyses, osc_freq))


class FrequencyShiftClosureAnalysis(LinkedCurveAnalysis):
    def __init__(self, analyses: List[RamseyXYAnalysis], osc_freq: float):
        super().__init__(
            analyses,
            linked_params={
                'amp': None,
                'tau': None,
                'freq': f'closure_factor * frequency_shift + {osc_freq}',
                'base': None
            },
            fixed_params={
                'tau': np.inf
            },
            experiment_params=['frequency_shift']
        )

        self.set_options(
            result_parameters=['closure_factor'],
            p0={'amp': 0.5, 'base': 0.5, 'closure_factor': -1.}
        )
