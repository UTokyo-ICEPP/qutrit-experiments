"""Generic circuit runner experiment."""
from collections.abc import Sequence
from typing import Optional, Union
import lmfit
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.result import Counts
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import BaseAnalysis, BaseExperiment, ExperimentData, Options

from ..constants import DEFAULT_SHOTS
from ..transpilation import map_and_translate
from ..util.get_metadata import get_metadata


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
        series_names = []
        for circuit in circuits:
            try:
                if circuit.metadata['series'] not in series_names:
                    series_names.append(circuit.metadata['series'])
            except KeyError:
                pass

        if not series_names:
            series_names = None

        if not analysis:
            analysis = DataExtraction(series_names)

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        if isinstance(circuits, QuantumCircuit):
            self._circuits = [circuits]
        else:
            self._circuits = list(circuits)

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []

        for circuit in self._circuits:
            if not circuit.metadata:
                circuit.metadata = {}

            circuit.metadata['qubits'] = self.physical_qubits
            if 'xval' not in circuit.metadata:
                circuit.metadata['xval'] = 0.

            circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        if self.experiment_options.trivial_transpilation:
            return map_and_translate(self.circuits(), self.physical_qubits, self._backend)
        return super()._transpiled_circuits()

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        shots = self.run_options.get('shots', DEFAULT_SHOTS)

        counts = []
        for circuit in transpiled_circuits:
            key = '0' * circuit.num_clbits
            counts.append(Counts({key: shots}))

        return counts


class DataExtraction(curve.CurveAnalysis):
    """Extract xy data from ExperimentData."""
    def __init__(
        self,
        series_names: Optional[list[str]] = None,
        series_key: str = 'series',
        outcome: Optional[str] = None,
        composite_index: Optional[Union[int, Sequence[int]]] = None
    ):
        if series_names is None:
            series_names = ['circuits']

        super().__init__(models=[lmfit.models.ConstantModel(name=name) for name in series_names])

        self.set_options(
            return_data_points=True,
            return_fit_parameters=False,
            plot=False
        )
        if outcome:
            self.set_options(outcome=outcome)

        if series_names:
            self.set_options(
                data_subfit_map={name: {series_key: name} for name in series_names}
            )

        self.composite_index = composite_index

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        if self.composite_index is None:
            return

        # "Flatten" the metadata if the experiment is composite
        # (CurveAnalysis assumes x values and data filter keys to be in the top-level metadata)
        for datum in experiment_data.data():
            metadata = datum['metadata']
            child_metadata = get_metadata(metadata, self.composite_index)
            for key, value in child_metadata.items():
                if key not in metadata:
                    metadata[key] = value

    def _run_curve_fit(
        self,
        curve_data: curve.CurveData,
        models: list['lmfit.Model'],
    ) -> curve.CurveFitResult:
        return curve.CurveFitResult(
            success=False,
            x_data=curve_data.x,
            y_data=curve_data.y
        )
