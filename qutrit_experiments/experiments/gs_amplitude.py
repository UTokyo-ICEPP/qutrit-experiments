from typing import Iterable, Optional, Dict, List, Sequence, Union, Callable
import numpy as np
import numpy.polynomial as poly
import scipy.optimize as sciopt

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import Options, ExperimentData
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.database_service import ExperimentEntryNotFound

from ..common.util import PolynomialOrder
from ..common.framework_overrides import BatchExperiment
from ..common.linked_curve_analysis import LinkedCurveAnalysis
from .gs_rabi import GSRabi, GSRabiAnalysis

twopi = 2. * np.pi

class GSAmplitude(BatchExperiment):
    """Experiment to find the amplitude dependency of the Rabi frequency."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.amplitudes = np.linspace(0.1, 0.95, 20)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        rabi_init: Optional[Callable] = None,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        amp_param_name: str = 'amp',
        width_param_name: Optional[str] = None,
        analysis_poly_order: PolynomialOrder = [1],
        backend: Optional[Backend] = None
    ):
        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        if np.any(np.array(amplitudes_exp) < 0.):
            raise RuntimeError('Amplitudes should be positive')

        experiments = []
        analyses = []

        if rabi_init is None:
            rabi_init = GSRabi

        amp_param = schedule.get_parameters(amp_param_name)[0]
        for idx, amp in enumerate(amplitudes_exp):
            sched = schedule.assign_parameters({amp_param: amp}, inplace=False)

            exp = rabi_init(physical_qubits, sched, widths=widths, time_unit=time_unit,
                            experiment_index=idx, backend=backend)
            exp.extra_metadata[amp_param_name] = amp
            if width_param_name is not None:
                exp.set_experiment_options(param_name=width_param_name)

            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = GSAmplitudeAnalysis(analyses, analysis_poly_order, experiment_param=amp_param_name)

        super().__init__(experiments, backend=backend, analysis=analysis)

        self.set_experiment_options(schedule=schedule)

        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)


class GSAmplitudeAnalysis(LinkedCurveAnalysis):
    def __init__(
        self,
        analyses: List[GSRabiAnalysis],
        poly_order: PolynomialOrder,
        experiment_param: str = 'amp'
    ):
        for analysis in analyses:
            analysis.set_options(
                p0={'amp': 1., 'base': 0.},
                bounds={
                    'amp': (0.9, 1.1),
                    'base': (-0.1, 0.1),
                    'freq': (0., np.inf),
                    'phase': (0., twopi)
                }
            )

        self.poly_order = PolynomialOrder(poly_order)

        coeffs = ['0.'] * (self.poly_order.order + 1)
        for power in self.poly_order.powers:
            coeffs[power] = '{var}_c%d' % power
        coeff_expr = ', '.join(coeffs)

        freq_func = f'polynomial.polynomial.polyval({experiment_param}, [{coeff_expr.format(var="freq")}])'
        phase_func = f'polynomial.polynomial.polyval({experiment_param}, [{coeff_expr.format(var="phase")}])'

        super().__init__(
            analyses,
            linked_params={
                'amp': None,
                'base': None,
                'freq': freq_func,
                'phase': phase_func
            },
            experiment_params=[experiment_param]
        )

        self.set_options(
            p0={'amp': 1., 'base': 0.},
            bounds={'amp': (0.9, 1.1), 'base': (-0.1, 0.1)}
        )
