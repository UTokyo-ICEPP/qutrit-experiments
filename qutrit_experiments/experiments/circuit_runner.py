"""Generic circuit runner experiment."""
from collections.abc import Sequence
from typing import Optional, Union
import lmfit
from qiskit import QuantumCircuit
from qiskit.providers import Backend
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing.processor_library import get_processor
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.framework.containers import ArtifactData, FigureType

from ..transpilation import map_and_translate


class CircuitRunner(BaseExperiment):
    """Generic circuit runner experiment."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.trivial_transpilation = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]],
        backend: Optional[Backend] = None,
        analysis: Optional[BaseAnalysis] = None
    ):
        if not analysis:
            series_names = set(c.metadata.get('series') for c in circuits)
            if series_names == set([None]):
                series_names = None
            analysis = DataExtraction(series_names)

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        if isinstance(circuits, QuantumCircuit):
            self._circuits = [circuits]
        else:
            self._circuits = list(circuits)

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []

        for orig in self._circuits:
            circuit = orig.copy()
            if 'xval' not in circuit.metadata:
                circuit.metadata['xval'] = 0.
            circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        if self.experiment_options.trivial_transpilation:
            return map_and_translate(self.circuits(), self.physical_qubits, self._backend)
        return super()._transpiled_circuits()


class DataExtraction(curve.CurveAnalysis):
    """Extract xy data from ExperimentData."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.composite_index = None
        return options

    def __init__(
        self,
        series_names: Optional[list[str]] = None,
        series_key: str = 'series',
        outcome: Optional[str] = None
    ):
        if (model_names := series_names) is None:
            model_names = ['circuits']

        super().__init__(models=[lmfit.models.ConstantModel(name=name) for name in model_names])

        if outcome:
            self.set_options(outcome=outcome)
        if series_names:
            self.set_options(
                data_subfit_map={name: {series_key: name} for name in series_names}
            )

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[Union[AnalysisResultData, ArtifactData]], list[FigureType]]:
        self._initialize(experiment_data)
        table = self._format_data(self._run_data_processing(experiment_data.data()))
        return [ArtifactData(name='curve_data', data=table)], []

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        data_processor = self.options.data_processor or get_processor(experiment_data, self.options)

        if not data_processor.is_trained:
            data_processor.train(data=experiment_data.data())
        self.set_options(data_processor=data_processor)

        if (compidx := self.options.composite_index) is None:
            return

        if isinstance(compidx, int):
            compidx = [compidx]

        # "Flatten" the metadata if the experiment is composite
        # (CurveAnalysis assumes x values and data filter keys to be in the top-level metadata)
        for datum in experiment_data.data():
            child_metadata = datum['metadata']
            for idx in compidx:
                child_metadata = child_metadata['composite_metadata'][idx]

            for key, value in child_metadata.items():
                if key not in datum['metadata']:
                    datum['metadata'][key] = value
