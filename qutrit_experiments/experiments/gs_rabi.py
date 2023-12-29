"""Rabi experiment with fixed amplitude and varied tone length."""
from collections.abc import Iterable, Sequence
from typing import Optional, Union
import numpy as np
import scipy
import lmfit

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import BaseExperiment, ExperimentData, Options
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor, Probability

from ..constants import DEFAULT_SHOTS, RZ_SIGN
from ..data_processing.to_expectation import ToExpectation
from ..transpilation.layout_only import replace_calibration_and_metadata

twopi = 2. * np.pi


class GSRabi(BaseExperiment):
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
        options.dummy_components = None

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
        readout_qubit = self.physical_qubits[measured_qubit]
        initial_state = self.experiment_options.initial_state

        circuit = QuantumCircuit(len(self._physical_qubits), 1)

        if initial_state == 'x':
            # Rz(-π/2) - SX - Rz(π/2) (physical)
            # Qiskit thinks the pulse envelope is modulated with exp(-iωt) and therefore
            # ShiftPhase(φ) effects a physical Rz(-φ). The scheduler thus translates rz(-φ) as
            # ShiftPhase(φ).
            # However, in the actual backends the modulation is with exp(iωt), which means
            # ShiftPhase(φ) actually effects a physical Rz(φ). Then, to be physically consistent,
            # at the circuit level we need to use rz(-φ).
            # We also drop the first Rz because the initial state is z+.
            angle = RZ_SIGN * np.pi / 2. # Need physical Rz(π/2) -> ShiftPhase(-π/2)
            if self.experiment_options.invert_x_sign:
                angle *= -1. # Need physical Rz(π/2) -> ShiftPhase(π/2) -> circuit rz(-π/2)
            circuit.sx(measured_qubit)
            circuit.rz(angle, measured_qubit)
        elif initial_state == 'y':
            circuit.sx(measured_qubit)
            circuit.rz(-np.pi, measured_qubit)

        circuit.metadata = {
            "intial_state": initial_state,
            "meas_basis": self.experiment_options.meas_basis,
            "readout_qubits": [readout_qubit] # needed for ReadoutMitigation
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
            circuit.rz(RZ_SIGN * np.pi / 2., measured_qubit)
            circuit.sx(measured_qubit)
        elif self.experiment_options.meas_basis == 'y':
            # Sdg + Hadamard sans last rz
            circuit.sx(measured_qubit)

        circuit.measure(measured_qubit, 0)

        return circuit

    def circuits(self) -> list[QuantumCircuit]:
        schedule = self.experiment_options.schedule
        width = schedule.get_parameters(self.experiment_options.param_name)[0]

        ## Build the template

        gate = Gate(schedule.name, len(self._physical_qubits), [])

        template = self._pre_circuit()
        template.append(gate, template.qregs[0])
        template.compose(self._post_circuit(), inplace=True)

        if not hasattr(template, 'metadata'):
            template.metadata = {}
        template.metadata['qubits'] = self.physical_qubits

        if self.experiment_options.time_unit is None:
            dt = 1.
        else:
            dt = self.experiment_options.time_unit

        ## Assign the width values

        circuits = []

        for width_val in self.experiment_options.widths:
            gs_sched = schedule.assign_parameters({width: width_val}, inplace=False)

            circuit = template.copy()
            circuit.add_calibration(gate, self._physical_qubits, gs_sched)
            circuit.metadata['xval'] = width_val * dt

            circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                self.transpile_options.target)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        from qiskit_aer import AerSimulator # pylint: disable=import-outside-toplevel

        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        options = self.experiment_options
        measured_qubit = options.measured_logical_qubit

        hamiltonian_components = options.dummy_components
        if hamiltonian_components is None:
            # Ideally would like to scale this with the pulse amplitude but it's difficult
            # to extract it
            hamiltonian_components = np.array([-1857082.,  -1232100., -138360.])

        paulis = np.array([[[0., 1.], [1., 0.]],
                           [[0., -1.j], [1.j, 0.]],
                           [[1., 0.], [0., -1.]]])
        hamiltonian = np.tensordot(hamiltonian_components, paulis, (0, 0))

        t_offset = 12.
        t = (np.array(options.widths) + t_offset) * options.time_unit
        unitaries = scipy.linalg.expm(-1.j * hamiltonian[None, ...] * t[:, None, None])

        circuits = []

        for source, unitary in zip(self.circuits(), unitaries):
            circuit = QuantumCircuit(1, 1)

            for inst in source.data:
                if inst.operation.name == options.schedule.name:
                    circuit.unitary(unitary, 0)
                elif len(inst.qubits) == 1 and inst.qubits[0].index == measured_qubit:
                    if len(inst.clbits):
                        cargs = [0]
                    else:
                        cargs = None

                    circuit.append(inst.operation, qargs=[0], cargs=cargs)

            circuits.append(circuit)

        simulator = AerSimulator()
        return simulator.run(circuits, shots=shots).result().get_counts()


class GSRabiAnalysis(curve.OscillationAnalysis):
    r"""OscillationAnalysis specific to the GS Rabi experiment.

    curve.guess.frequency assumes :math:`y = A \cos (2\pi f x + \phi)` when the FFT-based
    guess fails. CR Rabi is rather slow, so FFT usually fails, and we therefore need to
    have the input probability values shifted down by 0.5.
    """
    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.outcome = None
        return options

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

        data_processor = DataProcessor('counts', [Probability('0'), ToExpectation()])
        # Next line is actually unnecessary since options.outcome is used only when constructing
        # a default DataProcessor
        self.options.outcome = '0'
        self.options.data_processor = data_processor

        self.plotter.set_figure_options(
            xlabel='Pulse width',
            #xval_unit='s', # unit set by subanalyses
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

        fixed_freq_sign = 'freq' in self.options.fixed_parameters

        if not fixed_freq_sign:
            bounds = user_opt.bounds['freq']
            if bounds is None:
                user_opt.bounds.set_if_empty(
                    freq=(-np.inf, np.inf)
                )
            else:
                if bounds[0] >= 0. or bounds[1] <= 0.:
                    fixed_freq_sign = True

        if fixed_freq_sign:
            phase_bounds = (0., twopi)
        else:
            phase_bounds = (0., np.pi)

        user_opt.bounds.set_if_empty(
            phase=phase_bounds
        )

        has_phase_guess = user_opt.p0['phase'] is not None

        user_opts = super()._generate_fit_guesses(user_opt, curve_data)

        if has_phase_guess:
            # OscillationAnalysis simply repeats the same guess with different phase offsets
            user_opts = user_opts[:1]

        additional_opts = []

        if fixed_freq_sign:
            # Add negative phase guesses (which are lacking in OscillationAnalysis)
            for phase_guess in np.linspace(-np.pi * 0.75, -np.pi * 0.25, 3):
                user_opt = user_opts[0].copy()
                user_opt.p0['phase'] = phase_guess
                additional_opts.append(user_opt)

        else:
            for positive_freq_opt in user_opts:
                negative_freq_opt = positive_freq_opt.copy()
                negative_freq_opt.p0['freq'] *= -1.
                additional_opts.append(negative_freq_opt)

        user_opts += additional_opts

        return user_opts


class GSRabiTrigSumAnalysis(curve.CurveAnalysis):
    """GSRabiAnalysis with a different parametrization.

    Frequency is assumed to be positive.
    """
    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.outcome = None
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

        data_processor = DataProcessor('counts', [Probability('0'), ToExpectation()])
        self.options.outcome = '0'
        self.options.data_processor = data_processor

        self.plotter.set_figure_options(
            xlabel='Pulse width',
            #xval_unit='s', # unit set by subanalyses
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
