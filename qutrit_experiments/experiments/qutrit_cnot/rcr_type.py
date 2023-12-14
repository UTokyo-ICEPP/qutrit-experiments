from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import Options, AnalysisResultData, ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import (DATA_ENTRY_PREFIX,
                                                                   PARAMS_ENTRY_PREFIX)

from ...common.framework_overrides import BatchExperiment, CompoundAnalysis
from ..cr_rabi import CRRabi


twopi = 2. * np.pi

class QutritRCR02Test(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for eidx, control_state in enumerate([0, 2]):
            exp = CRRabi(physical_qubits, schedule, control_state,
                         widths=widths, time_unit=time_unit, experiment_index=eidx, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = QutritRCR02TestAnalysis(analyses)
        super().__init__(experiments, backend=backend, analysis=analysis)


class QutritRCR02TestAnalysis(CompoundAnalysis):
    def __init__(self, analyses: List['GSRabiAnalysis']):
        super().__init__(analyses)
        for analysis in analyses:
            analysis.set_options(return_data_points=True)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Determine the type of RCR sequence to use.

        - Check the position & value of the first min
        - If min is not the last value, fit should have succeeded. Otherwise estimate the omega value.
        - Use the faster oscillation
        """
        component_index = experiment_data.metadata["component_child_index"]
        omega = {}
        for ichild, control_state in enumerate([0, 2]):
            child_data = experiment_data.child_data(component_index[ichild])
            cdata = child_data.analysis_results(DATA_ENTRY_PREFIX + self._analyses[ichild].name).value
            min_pos = np.amin(cdata['ydata'])
            if min_pos > cdata['ydata'].shape[0] * 0.9:
                phase_advance = np.arccos(cdata['ydata'][-1]) - np.arccos(cdata['ydata'][0])
                omega[control_state] = phase_advance / (cdata['xdata'][-1] - cdata['xdata'][0])
            else:
                omega[control_state] = child_data.analysis_results('freq') * twopi

        if np.abs(omega[0]) > np.abs(omega[2]):
            res = AnalysisResultData(name='rcr_type', value=0)
        else:
            res = AnalysisResultData(name='rcr_type', value=2)

        return [res], []


class QutritRCRTypeCal(BaseCalibrationExperiment, QutritRCR02Test):
    """Determine the RCR sequence type."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'rcr90',
        cal_parameter_name: List[str] = ['counter_amp', 'counter_phase'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        analysis_poly_order: PolynomialOrder = [1]
    ):
        assign_params = {
            'cr_amp': 0.,
            'cr_phase': 0.,
            'counter_amp': Parameter('amp'),
            'counter_stark_amp': 0.,
            'counter_phase': 0.,
            'cr_width': Parameter('width'),
            'cr_margin': 0.
        }
        cr_schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                                assign_params=assign_params, group=group)

        counter_pulse = next(block.pulse for block in cr_schedule.blocks
                             if block.name == 'Counter')

        with pulse.build(name='Counter') as schedule:
            pulse.play(counter_pulse, backend.drive_channel(physical_qubits[1]))

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            amplitudes=amplitudes,
            widths=widths,
            time_unit=time_unit,
            analysis_poly_order=analysis_poly_order
        )
