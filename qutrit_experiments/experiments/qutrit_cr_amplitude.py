from typing import Iterable, Optional, Sequence, List, Dict
import numpy as np
import lmfit

from qiskit import QuantumCircuit, pulse
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import Options, ExperimentData
import qiskit_experiments.curve_analysis as curve

from ..common.linked_curve_analysis import LinkedCurveAnalysis
from ..common.util import rabi_freq_per_amp, grounded_gauss_area, PolynomialOrder
from .gs_rabi import GSRabiAnalysis
from .gs_amplitude import GSAmplitude, GSAmplitudeAnalysis
from .cr_rabi import cr_rabi_init

twopi = 2. * np.pi


class QutritCRAmplitude(GSAmplitude):
    """Experiment to find the CR amplitude for generalized CNOT.

    This experiment performs a Rabi oscillation frequency measurement on the target qubit, changing
    the amplitude of the CR tone.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.control_state = None
        options.max_circuits = 150
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        schedule: ScheduleBlock,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        analysis_poly_order: PolynomialOrder = 5,
        backend: Optional[Backend] = None
    ):
        super().__init__(
            physical_qubits,
            schedule,
            rabi_init=cr_rabi_init(control_state),
            amplitudes=amplitudes,
            widths=widths,
            time_unit=time_unit,
            amp_param_name='cr_amp',
            analysis_poly_order=analysis_poly_order,
            backend=backend
        )

        # The schedule parameter can be the actual cr schedule or qutrit_cx
        def nested_search(block, pulse_name):
            for element in block.blocks:
                if isinstance(element, ScheduleBlock):
                    if (named_pulse := nested_search(element, pulse_name)) is not None:
                        return named_pulse
                elif isinstance(element, pulse.Play):
                    if element.name == pulse_name:
                        return element.pulse

            return None

        if (counter_pulse := nested_search(schedule, 'Counter')) is None:
            raise RuntimeError('Given schedule does not contain the counter pulse')

        counter_amp = np.abs(counter_pulse.amp)

        if counter_amp == 0.:
            cr_width = schedule.get_parameters('cr_width')[0]
            schedule_w0 = schedule.assign_parameters({cr_width: 0.}, inplace=False)
            cr_pulse = nested_search(schedule_w0, 'CR')
            cr_sigma = float(cr_pulse.sigma)
            try:
                cr_rsr = cr_pulse.support / (2. * cr_sigma) # ModulatedGaussianSquare
            except AttributeError:
                cr_rsr = cr_pulse.duration / (2. * cr_sigma)
            cr_flank_area = grounded_gauss_area(cr_sigma, cr_rsr, gs_factor=True)

            freq_asymptote = 1.e+6
            if backend:
                # CR entangling Rabi oscillation frequency is of the order of the interaction parameter
                # cf. Malekakhlagh et al. PRA 102 042605
                q1 = min(physical_qubits)
                q2 = max(physical_qubits)
                for prop in backend.properties().general:
                    if prop.name == f'jq_{q1}{q2}':
                        freq_asymptote = prop.value
                        if prop.unit == 'kHz':
                            freq_asymptote *= 1.e+3
                        elif prop.unit == 'MHz':
                            freq_asymptote *= 1.e+6
                        elif prop.unit == 'GHz':
                            freq_asymptote *= 1.e+9
                        break

            for exp, amp in zip(self.component_experiment(), self.experiment_options.amplitudes):
                # We need to give some fit hints because the frequency will be very low.
                # Assert the frequency to be positive, even though the corresponding Hamiltonian component is signed.
                # Sign of the component is to be revealed through an actual tomography
                freq = freq_asymptote * amp
                phase = freq * twopi * cr_flank_area * exp.experiment_options.time_unit

                exp.extra_metadata['fit_p0'].update({'freq': freq, 'phase': phase})

        self.set_experiment_options(control_state=control_state)

        self.analysis.options.fixed_parameters['amp'] = 1.
        for suba in self.analysis.component_analysis():
            suba.options.fixed_parameters['amp'] = 1.

        self.analysis.plotter.set_figure_options(ylim=(-1.1, 1.1))

        self.analysis.linked_param_figure_options = {
            'base': {'ylim': (-0.5, 0.5)},
            'phase': {'ylim': (0., twopi)}
        }

    def _metadata(self) -> Dict[str, any]:
        metadata = super()._metadata()
        metadata['control_state'] = self.experiment_options.control_state
        return metadata
