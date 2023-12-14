from typing import Tuple, List
import warnings

from qiskit_experiments.framework import ExperimentData, Options, AnalysisResultData
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...common.framework_overrides import CompoundAnalysis

class CRMultiSimultaneousFitAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def __init__(self, analyses):
        super().__init__(analyses, flatten_results=False)
        # Turn off subanalysis plotting off by default because LinkedCurveAnalysis can have tens of models, and the calculation
        # of the error bands (eval_with_uncertainties) can take several seconds per model.
        for analysis in analyses:
            analysis.options.plot = False

        self.plotter = CurvePlotter(MplDrawer())

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        # CompositeAnalysis with flatten_results=False replaces the sub-analysis
        # results within the parent container instead of returning them
        component_index = experiment_data.metadata["component_child_index"]
        child_data = [experiment_data.child_data(component_index[ichld]) for ichld in range(len(self._analyses))]

        fit_results = {}
        metadata = {}

        for analysis, data in zip(self._analyses, child_data):
            try:
                control_state = data.metadata['control_state']
            except KeyError:
                # backward compat
                control_state = analysis.options.extra['control_state']

            fit_result = data.analysis_results(PARAMS_ENTRY_PREFIX + analysis.name).value
            if not fit_result.success:
                warnings.warn(f'LinkedCurveAnalysis for control_state={control_state} failed', UserWarning)
                return [], []

            fit_results[control_state] = fit_result
            metadata[control_state] = data.metadata

        results = self._solve(fit_results, metadata)

        if self.options.plot:
            self._plot_curves(experiment_data)

            return results, [self.plotter.figure()]

        else:
            return results, []
