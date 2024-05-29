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

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        shots = self.run_options.get('shots', DEFAULT_SHOTS)

        counts = []
        for circuit in transpiled_circuits:
            key = '0' * circuit.num_clbits
            counts.append(Counts({key: shots}))

        return counts


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

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

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
