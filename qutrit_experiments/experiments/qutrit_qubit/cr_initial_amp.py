"""Initial CR amplitude determination.

Nonlinear effect in CR has been observed to break the block-diagonality. We check the decay of
control=|2> as a function of the amplitude and determine an acceptable amplitude.
"""
from collections.abc import Sequence
import logging
from typing import Optional, Union
import lmfit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.curve_data import ParameterRepr
from qiskit_experiments.framework import BaseExperiment, ExperimentData

from ...experiment_mixins import MapToPhysicalQubits
from ...framework.ternary_mcm_analysis import TernaryMCMResultAnalysis
from ...gates import CrossResonanceGate, X12Gate

logger = logging.getLogger(__name__)


class CRDiagonality(MapToPhysicalQubits, BaseExperiment):
    """CR characterization.

    Experiment: Raise the control qubit to |2>, run the CR tone, and measure the control state.
    Mid-circuit measurement is used to discriminate |2> from |0> and |1>:
    state result
    2     11
    1     01
    0     10
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.amplitudes = np.linspace(0., 0.9, 10)
        return options
    
    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.rep_delay = 5.e-4
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        amplitudes: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=CRDiagonalityAnalysis(), backend=backend)
        self.set_experiment_options(schedule=schedule)
        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)

    def circuits(self) -> list[QuantumCircuit]:
        sched = self.experiment_options.schedule
        amplitude = next(iter(sched.parameters))

        template = QuantumCircuit(2, 2)
        template.x(0)
        template.append(X12Gate(), [0])
        template.append(CrossResonanceGate([amplitude]), [0, 1])
        template.measure(0, 0)
        template.x(0)
        template.measure(0, 1)
        template.add_calibration('cr', self.physical_qubits, sched, [amplitude])

        circuits = []
        for aval in self.experiment_options.amplitudes:
            circ = template.assign_parameters({amplitude: aval}, inplace=False)
            circ.metadata = {'xval': aval}
            circuits.append(circ)

        return circuits


class CRDiagonalityAnalysis(TernaryMCMResultAnalysis):
    """Analysis for CRDiagonality."""
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr='c0 + 0 * x',
                    name='model0'
                ),
                lmfit.models.ExpressionModel(
                    expr='a * x ** 2 + b1',
                    name='model1'
                ),
                lmfit.models.ExpressionModel(
                    expr='-a * x ** 2 + b2',
                    name='model2',
                )
            ],
            name=name
        )
        self.set_options(
            result_parameters=[ParameterRepr('a', 'curvature')],
            bounds={'a': (0., np.inf)}
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        data0 = curve_data.get_subset_of('model0')
        data1 = curve_data.get_subset_of('model1')
        data2 = curve_data.get_subset_of('model2')
        bmax = np.max(data1.y + data2.y)
        user_opt.bounds.set_if_empty(b1=(0., bmax), b2=(0., bmax))
        p0_b1 = data1.y[0]
        p0_b2 = data2.y[0]
        p0_a = (data1.y[-1] - p0_b1) / data1.x[-1] ** 2
        p0_c0 = np.mean(data0.y)
        user_opt.p0.set_if_empty(a=p0_a, b1=p0_b1, b2=p0_b2, c0=p0_c0)

        return user_opt


class CRInitialAmplitudeCal(BaseCalibrationExperiment, CRDiagonality):
    """Calibration of CR amplitude using CRDiagonality."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'cr_amp',
        schedule_name: str = 'cr',
        auto_update: bool = True,
        cutoff: float = 0.99,
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[int] = None
    ):
        if width is None:
            width = 256

        assign_params = {
            cal_parameter_name: Parameter('amplitude'),
            'cr_stark_amp': 0.,
            'counter_amp': 0.,
            'counter_stark_amp': 0.,
            'width': width
        }
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amplitudes=amplitudes
        )
        self._cutoff = cutoff

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        curvature = BaseUpdater.get_value(experiment_data, 'curvature')
        cr_amp = min(0.8, np.sqrt((1. - self._cutoff) / curvature))
        BaseUpdater.add_parameter_value(self._cals, experiment_data, cr_amp, self._param_name,
                                        schedule=self._sched_name,
                                        group=self.experiment_options.group)
