"""Analysis for experiments with ternary results classified with MCM (measure-X-measure)."""
from typing import Optional
import lmfit
import pandas as pd
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor, MarginalizeCounts, Probability
from qiskit_experiments.framework import ExperimentData, Options

from ..data_processing.readout_mitigation import ReadoutMitigation


class TernaryMCMResultAnalysis(curve.CurveAnalysis):
    """Analysis for experiments with ternary results classified with MCM (measure-Xminus-measure).

    The processed data have a structure of x=[x0, x0, x0, x1, x1, x1, ...],
    y=[P0(0), P0(1), P0(2), P1(0), P1(1), P1(2), ...], data_allocation=[0, 1, 2, 0, 1, 2, ...]
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.mcm_cbits = None
        options.assignment_matrix = None
        return options

    def __init__(self, models: list[lmfit.Model], name: Optional[str] = None):
        super().__init__(models=models, name=name)
        self.set_options(
            # Dummy subfit map required when multiple models are present
            # See curve_analysis.py L173 (QE 0.5.4)
            data_subfit_map={model._name: {} for model in self._models}
        )

    def _initialize(self, experiment_data: ExperimentData):
        opts = self.options
        if opts.data_processor is None:
            nodes = []
            if opts.mcm_cbits is not None:
                nodes.append(MarginalizeCounts(set(opts.mcm_cbits)))
            if opts.assignment_matrix is not None:
                nodes.append(ReadoutMitigation(opts.assignment_matrix))
            nodes.append(Probability('00'))  # dummy outcome
            opts.data_processor = DataProcessor('counts', nodes)

        super()._initialize(experiment_data)

    def _run_data_processing(
        self,
        raw_data: list[dict],
        category: str = 'raw'
    ) -> curve.ScatterTable:
        dp = self.options.data_processor
        tables = []
        for series_id, (model, outcome) in enumerate(zip(self._models, ['10', '01', '11'])):
            dp._nodes[-1]._outcome = outcome
            table = super()._run_data_processing(raw_data, category=category)
            table.series_name = model._name
            table.series_id = series_id
            tables.append(table)

        return curve.ScatterTable.from_dataframe(
            pd.concat([t.dataframe for t in tables], ignore_index=True)
        )
