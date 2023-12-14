from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.figure import Figure
import numpy as np
import lmfit
from uncertainties import ufloat, unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.pulse.channels import PulseChannel
from qiskit.result import Counts
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations, ParameterValue
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.curve_analysis.utils import convert_lmfit_result, eval_with_uncertainties
from qiskit_experiments.framework import AnalysisResultData, BackendData, ExperimentData, Options
from qiskit_experiments.framework.matplotlib import default_figure_canvas
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..common.linked_curve_analysis import LinkedCurveAnalysis
from ..common.framework_overrides import BatchExperiment, CompoundAnalysis
from .sizzle import build_sizzle_schedule
from .delay_phase_offset import RamseyPhaseSweep

twopi = 2. * np.pi


class SiZZle1ZPhaseScan(BatchExperiment):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.phase_offsets = np.linspace(0., twopi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        amplitudes: Optional[Tuple[float, float]] = None,
        control_phase_offsets: Optional[Sequence[float]] = None,
        channels: Optional[Tuple[PulseChannel, PulseChannel]] = None,
        delay_durations: Optional[Sequence[int]] = None,
        num_points: Optional[int] = None,
        pre_delay: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        if (offset_values := control_phase_offsets) is None:
            offset_values = self._default_experiment_options().phase_offsets

        pre_schedule = (XGate(), [0])

        experiments = []
        analyses = []

        for iexp, offset in enumerate(offset_values):
            delay_schedule = build_sizzle_schedule(frequency, physical_qubits[0], physical_qubits[1],
                                                   backend, amplitudes=amplitudes,
                                                   control_phase_offset=offset, channels=channels,
                                                   pre_delay=pre_delay)

            exp = RamseyPhaseSweep(physical_qubits, delay_durations=delay_durations,
                                   delay_schedule=delay_schedule, num_points=num_points,
                                   pre_schedule=pre_schedule, target_logical_qubit=1,
                                   extra_metadata={'control_phase_offset': offset},
                                   experiment_index=iexp, backend=backend)

            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=SiZZle1ZPhaseScanAnalysis(analyses))

        if control_phase_offsets is not None:
            self.set_experiment_options(phase_offsets=control_phase_offsets)


class SiZZle1ZPhaseScanAnalysis(LinkedCurveAnalysis):
    def __init__(self, analyses: List['RamseyPhaseSweepAnalysis']):
        super().__init__(
            analyses,
            linked_params={
                'amp': None,
                'base': None,
                'omega_z_dt': 'omega_amp * cos(control_phase_offset - channel_phase_diff) + omega_base'
            },
            experiment_params=['control_phase_offset']
        )

        self.set_options(
            result_parameters=['channel_phase_diff']
        )
