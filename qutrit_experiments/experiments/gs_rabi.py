"""Rabi experiment with fixed amplitude and varied tone length."""
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union
import numpy as np
import lmfit

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import BaseExperiment, ExperimentData, Options
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability

from ..experiment_mixins import MapToPhysicalQubitsCommonCircuit

twopi = 2. * np.pi


class GSRabi(MapToPhysicalQubitsCommonCircuit, BaseExperiment):
    """Rabi experiment where the width parameter of a GaussianSquare pulse is varied."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()

        options.schedule = None
        options.widths = np.linspace(0., 1024., 17)
        options.measured_logical_qubit = 0
        options.initial_state = 'z'
        options.meas_basis = 'z'
        options.time_unit = None
        options.experiment_index = None
        options.param_name = 'width'

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        widths: Optional[Iterable[float]] = None,
        initial_state: Optional[str] = None,
        meas_basis: Optional[str] = None,
        time_unit: Optional[float] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, backend=backend)

        self.set_experiment_options(schedule=schedule)
        if widths is not None:
            self.set_experiment_options(widths=widths)

        if initial_state is not None:
            if initial_state in ['x', 'y', 'z']:
                self.set_experiment_options(initial_state=initial_state)
            else:
                raise ValueError(f'Invalid initial state {initial_state}')

        if meas_basis is not None:
            if meas_basis in ['x', 'y', 'z']:
                self.set_experiment_options(meas_basis=meas_basis)
            else:
                raise ValueError(f'Invalid measurement basis {meas_basis}')

        if time_unit is None and backend is not None:
            time_unit = self._backend_data.dt
        if time_unit is not None:
            self.set_experiment_options(time_unit=time_unit)

        if experiment_index is not None:
            self.set_experiment_options(experiment_index=experiment_index)

        self.analysis = GSRabiAnalysis()
        self.analysis.set_options(
            result_parameters=['freq']
        )

        self.extra_metadata = {'fit_p0': {}}

    @property
    def num_circuits(self) -> int:
        return len(self.experiment_options.widths)

    def _pre_circuit(self) -> QuantumCircuit:
        """To be overridden by subclasses."""
        measured_qubit = self.experiment_options.measured_logical_qubit
        initial_state = self.experiment_options.initial_state

        circuit = QuantumCircuit(len(self._physical_qubits), 1)

        if initial_state == 'x':
            circuit.sx(measured_qubit)
            circuit.rz(np.pi / 2., measured_qubit)
        elif initial_state == 'y':
            circuit.sx(measured_qubit)
            circuit.rz(-np.pi, measured_qubit)

        circuit.metadata = {
            "initial_state": initial_state,
            "meas_basis": self.experiment_options.meas_basis
        }
        if self.experiment_options.experiment_index is not None:
            circuit.metadata['experiment_index'] = self.experiment_options.experiment_index

        return circuit

    def _post_circuit(self) -> QuantumCircuit:
        """To be overridden by subclasses."""
        circuit = QuantumCircuit(len(self._physical_qubits), 1)

        measured_qubit = self.experiment_options.measured_logical_qubit

        if self.experiment_options.meas_basis == 'x':
            # Hadamard with global phase pi/4 sans last rz
            circuit.rz(np.pi / 2., measured_qubit)
            circuit.sx(measured_qubit)
        elif self.experiment_options.meas_basis == 'y':
            # Sdg + Hadamard sans last rz
            circuit.sx(measured_qubit)

        return circuit

    def circuits(self) -> list[QuantumCircuit]:
        schedule = self.experiment_options.schedule
        width = schedule.get_parameters(self.experiment_options.param_name)[0]

        # Build the template

        gate = Gate(schedule.name, len(self._physical_qubits), [])

        template = self._pre_circuit()
        template.barrier()
        template.append(gate, template.qregs[0])
        template.barrier()
        template.compose(self._post_circuit(), inplace=True)
        template.measure(self.experiment_options.measured_logical_qubit, 0)

        if self.experiment_options.time_unit is None:
            dt = 1.
        else:
            dt = self.experiment_options.time_unit

        # Assign the width values

        circuits = []

        for width_val in self.experiment_options.widths:
            gs_sched = schedule.assign_parameters({width: width_val}, inplace=False)

            circuit = template.copy()
            circuit.add_calibration(gate, self._physical_qubits, gs_sched)
            circuit.metadata['xval'] = width_val * dt

            circuits.append(circuit)

        return circuits

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata


class GSRabiAnalysis(curve.OscillationAnalysis):
    r"""OscillationAnalysis specific to the GS Rabi experiment.

    curve.guess.frequency assumes :math:`y = A \cos (2\pi f x + \phi)` when the FFT-based
    guess fails. CR Rabi is rather slow, so FFT usually fails, and we therefore need to
    have the input probability values shifted down by 0.5.
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.phase_offset = 0.
        return options

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.set_options(
            data_processor=DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        )
        self.plotter.set_figure_options(
            xlabel='Pulse width',
            ylabel='Pauli expectation',
            ylim=(-1.2, 1.2)
        )

    def _initialize(self, experiment_data: ExperimentData):
        super()._initialize(experiment_data)
        if 'fit_p0' in experiment_data.metadata:
            self.options.p0.update(experiment_data.metadata['fit_p0'])

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        frequency bounded -> phase [0, 2pi], if not fixed make sure p0 sign is consistent
        frequency unbounded -> phase [0, pi]

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.
        Returns:
            list of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            amp=(-1.e-3, 2.)
        )
        user_opt.p0.set_if_empty(
            amp=1.,
            base=0.
        )

        freq_sign = None
        if (freq_p0 := self.options.fixed_parameters.get('freq')) is not None:
            freq_sign = np.sign(freq_p0)
        elif (freq_bounds := self.options.bounds.get('freq')) is not None:
            if freq_bounds[0] >= 0.:
                freq_sign = 1.
            elif freq_bounds[1] <= 0.:
                freq_sign = -1.

        if freq_sign == 1.:
            user_opt.bounds.set_if_empty(phase=(0., twopi))
        elif freq_sign == -1.:
            user_opt.bounds.set_if_empty(phase=(-twopi, 0.))

        # superclass options are copies of options with only the phase p0 varied; we'll make our own
        # p0s
        user_opt = super()._generate_fit_guesses(user_opt, curve_data)[0]

        if 'phase' in self.options.p0:
            return user_opt

        user_opts = []

        if freq_sign is None:
            for phase_guess in np.linspace(-np.pi, np.pi, 9):
                if phase_guess == 0.:
                    continue
                new_opt = user_opt.copy()
                if freq_bounds is None:
                    new_opt.bounds['freq'] = (0., np.inf) if phase_guess > 0. else (-np.inf, 0.)
                new_opt.bounds['phase'] = (0., np.pi) if phase_guess > 0. else (-np.pi, 0.)
                new_opt.p0['freq'] = np.sign(phase_guess) * np.abs(user_opt.p0['freq'])
                new_opt.p0['phase'] = phase_guess
                user_opts.append(new_opt)
        else:
            for phase_guess in np.linspace(0., twopi, 8, endpoint=False) * freq_sign:
                new_opt = user_opt.copy()
                new_opt.p0['phase'] = phase_guess
                user_opts.append(new_opt)

        for opt in user_opts:
            opt.bounds['phase'] = tuple(np.array(opt.bounds['phase']) + self.options.phase_offset)
            opt.p0['phase'] += self.options.phase_offset

        return user_opts

    def _run_curve_fit(
        self,
        curve_data: curve.CurveData,
        models: list[lmfit.Model],
    ) -> curve.CurveFitResult:
        current = np.geterr()['invalid']
        np.seterr(invalid='ignore')
        result = super()._run_curve_fit(curve_data, models)
        np.seterr(invalid=current)
        return result


class GSRabiTrigSumAnalysis(curve.CurveAnalysis):
    """GSRabiAnalysis with a different parametrization.

    Frequency is assumed to be positive.
    """
    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        return options

    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="alpha * cos(2 * pi * freq * x) + beta * sin(2 * pi * freq * x) + base",
                    name="trigsum",
                )
            ],
            name=name,
        )
        self.set_options(
            data_processor=DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        )
        self.plotter.set_figure_options(
            xlabel='Pulse width',
            ylabel='Pauli expectation',
            ylim=(-1.2, 1.2)
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.
        Returns:
            list of fit options that are passed to the fitter function.
        """
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            alpha=(-1., 1.),
            beta=(-1., 1.),
            base=(-max_abs_y, max_abs_y),
            freq=(0., np.inf)
        )
        user_opt.p0.set_if_empty(
            freq=curve.guess.frequency(curve_data.x, curve_data.y),
            base=curve.guess.constant_sinusoidal_offset(curve_data.y),
        )

        amp = curve.guess.max_height(curve_data.y - user_opt.p0["base"], absolute=True)[0]

        options = []
        for phase_guess in np.linspace(-np.pi * 0.75, np.pi, 8):
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(
                alpha=(amp * np.cos(phase_guess)),
                beta=(-amp * np.sin(phase_guess))
            )
            options.append(new_opt)

        return options
