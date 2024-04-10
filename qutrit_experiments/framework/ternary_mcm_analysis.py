"""Analysis for experiments with ternary results classified with MCM (measure-X-measure)."""
import lmfit
import numpy as np
from typing import Optional
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.framework import ExperimentData, Options

from ..data_processing import MultiProbability, ReadoutMitigation, SerializeMultiProbability

class TernaryMCMResultAnalysis(curve.CurveAnalysis):
    """Analysis for experiments with ternary results classified with MCM (measure-X-measure).
    
    The processed data have a structure of x=[x0, x0, x0, x1, x1, x1, ...],
    y=[P0(0), P0(1), P0(2), P1(0), P1(1), P1(2), ...], data_allocation=[0, 1, 2, 0, 1, 2, ...]
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.assignment_matrix = None
        return options
    
    def __init__(self, models: Optional[list[lmfit.Model]] = None, name: Optional[str] = None):
        super().__init__(models=models, name=name)
        self.set_options(
            # Dummy subfit map required when multiple models are present
            # See curve_analysis.py L173 (QE 0.5.4)
            data_subfit_map={model._name: {} for model in self._models}
        )

    def _initialize(self, experiment_data: ExperimentData):
        if self.options.data_processor is None:
            self.options.data_processor = self._make_data_processor()

        super()._initialize(experiment_data)

    def _run_data_processing(
        self,
        raw_data: list[dict],
        models: list[lmfit.Model]
    ) -> curve.CurveData:
        curve_data = super()._run_data_processing(raw_data, models)
        # ydata is serialized multiprobability -> repeated threefold
        return curve.CurveData(
            x=np.repeat(curve_data.x, 3),
            y=curve_data.y,
            y_err=curve_data.y_err,
            shots=np.repeat(curve_data.shots, 3),
            data_allocation=np.tile([0, 1, 2], curve_data.x.shape[0]),
            labels=curve_data.labels
        )

    def _make_data_processor(self) -> DataProcessor:
        nodes = []
        if (cal_matrix := self.options.assignment_matrix) is not None:
            nodes.append(ReadoutMitigation(cal_matrix))
        nodes += [
            MultiProbability(),
            SerializeMultiProbability(['10', '01', '11'])
        ]
        return DataProcessor('counts', nodes)
