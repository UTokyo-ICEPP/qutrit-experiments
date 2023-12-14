from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from uncertainties import ufloat
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations, BaseCalibrationExperiment
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

from ...common.framework_overrides import CompoundAnalysis, CompositeAnalysis, BatchExperiment
from ...common.util import get_cr_margin
from ..cr_rabi import cr_rabi_init
from ..hamiltonian_tomography import HamiltonianTomography


twopi = 2. * np.pi

class QutritRCR90Width(BatchExperiment):
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
            exp = HamiltonianTomography(physical_qubits, schedule,
                                        rabi_init=cr_rabi_init(control_state),
                                        widths=widths, time_unit=time_unit, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=QutritRCR90WidthAnalysis(analyses))


class QutritRCR90WidthAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Compute the rotary frequency to align the Z/X ratios."""
        component_index = experiment_data.metadata["component_child_index"]

        popts = {}
        omegas = {}
        for control_state, idx in zip([0, 2], component_index):
            analysis_results = experiment_data.child_data(idx).analysis_results()
            popts[control_state] = next(res.value.ufloat_params for res in analysis_results
                                        if res.name.startswith(PARAMS_ENTRY_PREFIX))
            omegas[control_state] = next(2. * res.value for res in analysis_results
                                         if res.name == 'hamiltonian_components')

        # Find the t such that
        #  (omega_0 * t + delta_0) - (omega_2 * t + delta_2) = ±π/2
        freq_diff = omegas[0][0] - omegas[2][0]
        offset_diff = popts[0]['delta'] - popts[2]['delta']
        signed_pi = np.sign(freq_diff.n) * np.pi
        rcr_rabi_diff = signed_pi / 2.
        cr_width = (rcr_rabi_diff - offset_diff) / freq_diff
        while cr_width < 0.:
            rcr_rabi_diff += signed_pi
            cr_width += signed_pi / freq_diff

        rabi_phase_2 = twopi * popts[2]['freq'] * cr_width + popts[2]['delta']

        analysis_results += [
            AnalysisResultData(name='cr_width_s', value=cr_width),
            AnalysisResultData(name='crcr_rabi_offset', value=rabi_phase_2),
            AnalysisResultData(name='rcr_rabi_diff', value=ufloat(rcr_rabi_diff, 0.))
        ]

        return analysis_results, figures


class QutritRCR90WidthCal(BaseCalibrationExperiment, QutritRCR90Width):
    """CR width determination for repeated CR with Rabi phase difference ±π/2."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: List[str] = ['cr', 'cr', 'crcr_rabi_offset', 'qutrit_cx'],
        cal_parameter_name: List[str] = ['cr_width', 'cr_margin', 'crcr_rabi_offset', 'rcr_rabi_diff'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None
    ):
        physical_qubits = tuple(physical_qubits)

        assign_params = {
            (cal_parameter_name[0], physical_qubits, schedule_name[0]): Parameter('cr_width'),
            (cal_parameter_name[1], physical_qubits, schedule_name[1]): 0.
        }
        schedule = calibrations.get_schedule('rcr90', physical_qubits,
                                             assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=sname,
                group=self.experiment_options.group,
            ) for pname, sname in zip(self._param_name, self._sched_name)
        ]

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        group = self.experiment_options.group
        result_index = self.experiment_options.result_index

        cr_width_s = BaseUpdater.get_value(experiment_data, 'cr_width_s',
                                           index=result_index)
        cr_width = cr_width_s / self._backend_data.dt
        cr_margin = get_cr_margin(cr_width, self.backend, self._cals, self.physical_qubits,
                                  self._sched_name[0])

        for pname, sname, value in zip(self._param_name[:2], self._sched_name[:2], [cr_width, cr_margin]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=sname, group=group)

        for pname, sname in zip(self._param_name[2:], self._sched_name[2:]):
            BaseUpdater.update(self._cals, experiment_data, pname, sname,
                               group=group, fit_parameter=pname)
