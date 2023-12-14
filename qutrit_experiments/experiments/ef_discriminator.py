from typing import Optional, Iterable, List, Tuple, Dict, Sequence
import numpy as np
import uncertainties.unumpy as unp
from sklearn.cluster import KMeans
try:
    import torch
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.framework import Options, ExperimentData, AnalysisResultData
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.visualization import CurvePlotter, MplDrawer
import qiskit_experiments.curve_analysis as curve

from ..common.ef_space import EFSpaceExperiment
from ..common.framework_overrides import BatchExperiment
from ..common.linked_curve_analysis import LinkedCurveAnalysis
from ..common.util import default_shots
from .gs_rabi import GSRabi
from .rabi import Rabi
from .rough_amplitude import EFRoughXSXAmplitudeCal
from .dummy_data import ef_memory


class EFDiscriminatorBase(EFSpaceExperiment):
    """Run a 1<->2 Rabi experiment and determine the optimal separation between the states in the IQ plane."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.separate_02 = True
        options.experiment_index = None
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.meas_level = MeasLevel.KERNELED
        options.meas_return = MeasReturnType.SINGLE
        return options

    def __init__(
        self,
        separate_02: bool = True,
        fine_boundary_optimization: bool = True,
        experiment_index: Optional[int] = None
    ):
        self.set_experiment_options(separate_02=separate_02)
        if not separate_02:
            self._final_xgate = False

        if experiment_index is not None:
            self.set_experiment_options(experiment_index=experiment_index)

        self.analysis = EFDiscriminatorAnalysis()
        if not fine_boundary_optimization:
            self.analysis.options.maxiter = 0

    def _pre_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1)
        circuit.metadata = {}
        if self.experiment_options.experiment_index is not None:
            circuit.metadata['experiment_index'] = self.experiment_options.experiment_index

        return circuit


class EFDiscriminator(EFDiscriminatorBase, GSRabi):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        separate_02: bool = True,
        fine_boundary_optimization: bool = True,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        GSRabi.__init__(self, physical_qubits, schedule, widths=widths, time_unit=time_unit, backend=backend)
        EFDiscriminatorBase.__init__(self, separate_02=separate_02,
                                     fine_boundary_optimization=fine_boundary_optimization,
                                     experiment_index=experiment_index)

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[np.ndarray]:
        widths = self.experiment_options.widths
        shots = self.run_options.get('shots', default_shots)
        one_probs = np.cos(2. * np.pi * 4.e-4 * widths + 0.2) * 0.49 + 0.51
        num_qubits = 1

        if self.experiment_options.separate_02:
            states = (0, 2)
        else:
            states = (1, 2)

        return ef_memory(one_probs, shots, num_qubits, states=states)


class EFDiscriminatorAmpBased(EFDiscriminatorBase, Rabi):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(-0.4, 0.4, 17)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        separate_02: bool = True,
        fine_boundary_optimization: bool = True,
        amplitudes: Optional[Iterable[float]] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        Rabi.__init__(self, physical_qubits, schedule, amplitudes=amplitudes, backend=backend)
        EFDiscriminatorBase.__init__(self, separate_02=separate_02,
                                     fine_boundary_optimization=fine_boundary_optimization,
                                     experiment_index=experiment_index)


class EFRoughXSXAmplitudeAndDiscriminatorCal(EFRoughXSXAmplitudeCal):
    def __init__(
        self,
        *args,
        fine_boundary_optimization: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.analysis = EFDiscriminatorAnalysis()
        if not fine_boundary_optimization:
            self.analysis.options.maxiter = 0

        if isinstance(self._param_name, str):
            self._param_name = [self._param_name, 'theta', 'dist']
            self._sched_name = [self._sched_name, 'iq_discriminator', 'iq_discriminator']
        else:
            self._param_name.extend(['theta', 'dist'])
            if isinstance(self._sched_name, str):
                self._sched_name = [self._sched_name, 'iq_discriminator', 'iq_discriminator']
            else:
                self._sched_name.extend(['iq_discriminator'] * 2)

    def update_calibrations(self, experiment_data: ExperimentData):
        super().update_calibrations(experiment_data)

        boundary = experiment_data.analysis_results('boundary', block=False).value

        for pname in ['theta', 'dist']:
            BaseUpdater.add_parameter_value(self._cals, experiment_data, boundary[pname], pname,
                                            schedule='iq_discriminator',
                                            group=self.experiment_options.group)

if HAS_TORCH:
    class ClassifiedOscillationAmp(torch.nn.Module):
        r"""Return the negative of the best-fit oscillation amp.

        Denote the GS tone width and measured IQ-plane data points for circuit :math:`j` as :math:`w_j`
        and :math:`\vec{x}_{jk} = (i_{jk}, q_{jk})^T` (:math:`j = 0, \dots, N_{\text{circ}} - 1`,
        :math:`k = 0, \dots, N_{\text{shot}} - 1`) respectively. The discriminator boundary is defined by
        :math:`\vec{x} \cdot \vec{n} = d`, where the norm vector :math:`\vec{n} := (\cos\theta, \sin\theta)^T`
        points towards the IQ region corresponding to the classification result :math:`|2\rangle` and
        :math:`d` is the signed distance of the classification boundary from the origin.

        The classification result (:math:`|1\rangle \rightarrow 0` and :math:`|2\rangle \rightarrow 1`)
        for each data point is approximated as

        .. math::

            y_{jk} = \frac{1}{2} \mathrm{tanh}(s[\vec{x}_{jk} \cdot \vec{n} - d]) + \frac{1}{2}

        where :math:`s` is a hyperparameter representing a scale factor. Then, for a given :math:`\theta` and
        :math:`d`, the expectation value of :math:`Z_{12}` for circuit :math:`j` is

        .. math::

            z_j = -\frac{1}{N_{\text{shot}}} \sum_{k} \mathrm{tanh}(s[\vec{x}_{jk} \cdot \vec{n} - d])

        The value of :math:`z_j` should oscillate with respect to :math:`w_j` with the frequency and phase
        offset already obtained from the OscillationAnalysis fit to the SVD-projected raw IQ data:

        .. math::

            c_j = \cos (2 \pi f w_j + \phi)

        An oscillation curve fit to :math:`z_j` amounts to minimizing

        .. math::

            L(a, b; \theta, d) = \sum_j (a c_j + b - z_j)^2

        for some amplitude :math:`a` and baseline offset :math:`b`. The best-fit parameters :math:`\hat{a}(\theta, d)`
        and :math:`\hat{b}(\theta, d)` are given by solving

        .. math::

            \frac{\partial L}{\partial a} = 0 \\
            \frac{\partial L}{\partial b} = 0,

        which translates to

        .. math::

            \begin{pmatrix} \sum_j c_j & N_{\mathrm{circ}} \\ \sum_j c_j^2 & \sum_j c_j \end{pmatrix}
            \begin{pmatrix} \hat{a} \\ \hat{b} \end{pmatrix}
            = \begin{pmatrix} \sum_j z_j \\ \sum_j c_j z_j \end{pmatrix}.

        We seek to find a classification boundary that maximizes :math:`\hat{a}`.
        """

        def __init__(
            self,
            freq: float,
            phase: float,
            xval: np.ndarray,
            theta: float = 0.,
            dist: float = 0.,
            dist_scale: float = 1.e+6
        ):
            super().__init__()

            self._theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.double))
            self._dist = torch.nn.Parameter(torch.tensor(dist / dist_scale, dtype=torch.double))
            mean = np.cos(2. * np.pi * freq * xval + phase)
            sumc = np.sum(mean)
            sumc2 = np.sum(np.square(mean))
            self.curve = torch.tensor(mean)
            self.sol_coeffs = np.linalg.inv([[sumc, mean.shape[0]], [sumc2, sumc]])[0]
            self.dist_scale = dist_scale

        @property
        def theta(self):
            return self._theta.item()

        @property
        def dist(self):
            return self._dist.item() * self.dist_scale

        def forward(self, x):
            """Forward pass.

            Args:
                x: Data points. Shape [N_circ, N_shot, 2]
            """
            vnorm = torch.tensor([torch.cos(self._theta), torch.sin(self._theta)])
            z = -torch.mean(torch.tanh(torch.inner(x / self.dist_scale, vnorm) - self._dist), 1)
            ahat = self.sol_coeffs[0] * torch.sum(z) + self.sol_coeffs[1] * torch.sum(self.curve * z)
            return -ahat

else:
    ClassifiedOscillationAmp = None

def fit_boundary(
    xval: np.ndarray,
    samples: np.ndarray,
    freq: float,
    phase: float,
    idx_i1: int,
    maxiter: int,
    convergence: float
) -> Tuple[Dict[str, float], np.ndarray]:
    """Perform a ClassifiedOscillationAmp fit to the experiment data.

    Args:
        xval: [N_circ]
        samples: [N_circ, N_shot, 1, 2]
        freq: Best-fit frequency.
        phase: Best-fit phase.
        idx_i1: The outermost index of the samples that should be the closest to i1.
        maxiter: Maximum number of SGD iterations. Set to 0 if K-means clusterization is sufficient.
        convergence: Loss convergence tolerance.
    """
    samples = np.squeeze(samples)

#     # Get initial guesses for theta and dist from KMeans
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(samples.reshape(-1, 2))
#     centers = kmeans.cluster_centers_

    # KMeans causes segfaults, so find the initial guesses by hand
    n_circuits = samples.shape[0]
    circuit_means = np.mean(samples, axis=1) # [N_circ, 2]
    dist2_matrix = np.sum(np.square(circuit_means[None, ...] - circuit_means[:, None, ...]), axis=-1)
    farthest_circuits = np.unravel_index(np.argmax(dist2_matrix), (n_circuits, n_circuits))
    centers = circuit_means[np.array(farthest_circuits)]

    # Which center is closer to samples[0]?
    d = samples[idx_i1][:, None, :] - centers[None, :, :]
    distmean = np.mean(np.sqrt(np.sum(np.square(d), axis=-1)), axis=0)
    i1 = np.argmin(distmean)
    i2 = np.argmax(distmean)

    # The norm vector points from i1 to i2
    i1toi2 = centers[i2] - centers[i1]
    centroids_dist = np.sqrt(np.dot(i1toi2, i1toi2))

    vnorm = i1toi2 / centroids_dist
    theta = np.arctan2(vnorm[1], vnorm[0])

    midpoint = np.mean(centers, axis=0)
    dist = np.dot(vnorm, midpoint)

    if maxiter <= 0:
        return {'theta': theta, 'dist': dist}, None

    # Because the data points go through SVD projection, sign of the y values are meaningless.
    # The target curve of the fit should correspond to <Z12>, so we adjust the phase to set
    # y[idx_i1] = cos(2pi * freq * xval[idx_i1] + phase) > 0.
    if np.cos(2. * np.pi * freq * xval[idx_i1] + phase) < 0.:
        phase += np.pi

    model = ClassifiedOscillationAmp(freq, phase, xval,
                                     theta=theta, dist=dist,
                                     dist_scale=(centroids_dist / 2.))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.e-2)

    losses = np.empty(maxiter)

    x = torch.squeeze(torch.from_numpy(samples))
    for istep in range(maxiter):
        loss = model(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[istep] = loss.item()

        if istep > 10 and np.amax(losses[istep - 9:istep + 1]) - np.amin(losses[istep - 9:istep + 1]) < convergence:
            break

    theta = model.theta
    dist = model.dist

    losses = losses[:istep + 1]

    while theta > np.pi:
        theta -= 2. * np.pi
    while theta < -np.pi:
        theta += 2. * np.pi

    return {'theta': theta, 'dist': dist}, losses


def draw_loss(
    losses: np.ndarray,
    title: Optional[str] = None
) -> 'pyplot.Figure':
    plotter = CurvePlotter(MplDrawer())
    plotter.set_figure_options(
        xlabel='iteration',
        ylabel='loss',
        figure_title=title
    )

    xval = np.arange(losses.shape[0])
    plotter.set_series_data(
        'loss',
        x_interp=xval,
        y_interp=losses
    )

    return plotter.figure()


def draw_iq_boundary(
    xval: np.ndarray,
    samples: np.ndarray,
    boundary: Dict[str, float],
    title: Optional[str] = None
) -> 'pyplot.Figure':
    nw = xval.shape[0]

    #TODO REPLACE WITH IQPLOTTER

    drawer = curve.MplCurveDrawer()
    nx = np.ceil(np.sqrt(nw)).astype(int)
    ny = np.ceil(nw / nx).astype(int)
    drawer.options.subplots = (ny, nx)
    drawer.options.xlabel = 'I'
    drawer.options.ylabel = 'Q'
    for iw in range(nw):
        drawer.options.plot_options[f'plot{iw}'] = {'canvas': iw}
    drawer.options.figure_title = title

    drawer.initialize_canvas()

    theta = boundary['theta']
    dist = boundary['dist']

    # Determine the boundary xy values to draw
    coms = np.mean(samples[:, :, 0, :], axis=1) # shot means for each circuit
    dists = np.sum(np.square(coms[:, None, :] - coms[None, :, :]), axis=-1) # circuit-circuit distances
    farthest_indices = np.unravel_index(np.argmax(dists), dists.shape) # farthest two circuits
    approx_cluster_centers = coms[np.array(farthest_indices)] # farthest shot means
    cluster_edge = np.diff(approx_cluster_centers, axis=0)[0] # vector connecting the cluster centers
    cluster_edge_perp = np.array([-cluster_edge[1], cluster_edge[0]]) # perpendicular vector to cluster_edge
    b_norm = np.array([np.cos(theta), np.sin(theta)]) # norm vector of the boundary
    # solve
    #  b_norm . x = dist
    #  cluster_edge_perp . (x - approx_cluster_centers[0]) = 0
    b_center = np.linalg.inv([b_norm, cluster_edge_perp]) @ np.array([dist, cluster_edge_perp @ approx_cluster_centers[0]])
    # boundary line vector
    b_dir = np.array([-np.sin(theta), np.cos(theta)])

    b_len = min(np.amax(samples[..., 0]) - np.amin(samples[..., 0]),
                np.amax(samples[..., 1]) - np.amin(samples[..., 1]))
    endpoints = b_center[:, None] + b_dir[:, None] * np.array([1., -1.]) * b_len / 2.

    for iw in np.argsort(xval):
        idata = samples[iw, :, 0, 0]
        qdata = samples[iw, :, 0, 1]
        drawer.draw_raw_data(idata, qdata, name=f'plot{iw}', s=0.01, marker='.')
        drawer.draw_fit_line(endpoints[0], endpoints[1], color='blue', name=f'plot{iw}')

    drawer.format_canvas()

    return drawer.figure


def ef_discriminator_analysis(
    experiment_data: ExperimentData,
    idx_i1: int,
    analysis_results: Optional[List[AnalysisResultData]] = None,
    maxiter: int = 10000,
    convergence: float = 1.e-4
) -> Tuple[Dict[str, float], np.ndarray, "pyplot.Figure", "pyplot.Figure"]:
    if analysis_results is None:
        analysis_results = experiment_data.analysis_results()

    fit_results = next(res for res in analysis_results if res.name.startswith(PARAMS_ENTRY_PREFIX))
    popt = fit_results.value.params

    xval = np.array(list(d['metadata']['xval'] for d in experiment_data.data()))
    samples = unp.nominal_values(DataProcessor('memory')(experiment_data.data()))

    sort_indices = np.argsort(xval)
    xval = xval[sort_indices]
    samples = samples[sort_indices]

    boundary, losses = fit_boundary(xval, samples, popt['freq'], popt['phase'], idx_i1, maxiter, convergence)

    if losses is not None:
        fig_loss = draw_loss(losses)
    else:
        fig_loss = None

    fig_iq = draw_iq_boundary(xval, samples, boundary)

    return boundary, losses, fig_loss, fig_iq


class EFDiscriminatorAnalysis(curve.OscillationAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.idx_i1 = np.argmin(np.abs(EFDiscriminatorAmpBased._default_experiment_options().amplitudes))
        options.convergence = 1.e-4
        options.maxiter = 10000
        return options

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.set_options(
            result_parameters=[curve.ParameterRepr("freq", "rabi_rate_12")],
            normalization=True,
        )
        self.plotter.set_figure_options(
            xlabel="Amplitude",
            ylabel="Signal (arb. units)",
        )

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:
        """Run the oscillation analysis and use the best-fit params to determine the discriminator."""
        if self.options.idx_i1 is None:
            raise AnalysisError('idx_i1 option must be set')

        analysis_results, figures = super()._run_analysis(experiment_data)

        boundary, losses, fig_loss, fig_iq = ef_discriminator_analysis(experiment_data,
                                                                       self.options.idx_i1,
                                                                       analysis_results=analysis_results,
                                                                       maxiter=self.options.maxiter,
                                                                       convergence=self.options.convergence)
        analysis_results.append(AnalysisResultData(name='boundary', value=boundary))
        if losses is not None:
            analysis_results.append(AnalysisResultData(name='loss', value=losses))
            figures.append(fig_loss)

        figures.append(fig_iq)

        return analysis_results, figures


class EFDiscriminatorMeasLOScan(BatchExperiment):
    """Perform the EFDiscriminator identification scanning over the measurement LO frequencies."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.freq_offsets = np.linspace(-0.2e+6, 0.2e+6, 15)
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = MeasReturnType.SINGLE

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        measure_schedule: ScheduleBlock,
        separate_02: bool = True,
        widths: Optional[Iterable[float]] = None,
        freq_offsets: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        freq_offsets_exp = freq_offsets
        if freq_offsets_exp is None:
            freq_offsets_exp = self._default_experiment_options().freq_offsets

        experiments = list()

        freq_offset = Parameter('freq_offset')
        with pulse.build(name='measure') as measure_sched:
            pulse.shift_frequency(freq_offset, pulse.MeasureChannel(physical_qubits[0]))
            pulse.call(measure_schedule)

        # Remap the memory slots (otherwise the calibration will be ignored)
        old_acquire = next(inst for _, inst in measure_sched.instructions
                           if isinstance(inst, pulse.Acquire) and
                              inst.channel == pulse.AcquireChannel(physical_qubits[0]))
        new_acquire = pulse.Acquire(old_acquire.duration, old_acquire.channel, pulse.MemorySlot(0))
        measure_sched.replace(old_acquire, new_acquire, inplace=True)

        self.measure_schedules = list(measure_sched.assign_parameters({freq_offset: offset}, inplace=False)
                                      for offset in freq_offsets_exp)

        for idx, offset in enumerate(freq_offsets_exp):
            exp = EFDiscriminator(physical_qubits, schedule, separate_02=separate_02, widths=widths,
                                  time_unit=time_unit, experiment_index=idx, backend=backend)
            experiments.append(exp)

        super().__init__(experiments, backend=backend)

        self.analysis = EFDiscriminatorMeasLOScanAnalysis(freq_offsets_exp)

    def circuits(self) -> List[QuantumCircuit]:
        circs = super().circuits()
        for circuit in circs:
            meas_sched = self.measure_schedules[circuit.metadata['composite_index'][0]]
            circuit.add_calibration('measure', [self.physical_qubits[0]], meas_sched)

        return circs


class EFDiscriminatorMeasLOScanAnalysis(LinkedCurveAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.convergence = 1.e-4
        options.maxiter = 10000
        return options

    def __init__(
        self,
        freq_offsets: Sequence[float],
        name: Optional[str] = None
    ):
        analyses = list(curve.OscillationAnalysis() for _ in range(len(freq_offsets)))
        super().__init__(
            analyses,
            linked_params={'freq': None, 'phase': None},
            name=name
        )

        self.freq_offsets = freq_offsets

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Run the individual analyses and plot the Rabi amplitude against the frequency offset."""
        # Batch experiments don't set their own meas_level and meas_return metadata, which makes
        # BaseCurveAnalysis (get_processor to be precise) assume that they are CLASSIFIED/avg
        experiment_data.metadata['meas_level'] = MeasLevel.KERNELED
        experiment_data.metadata['meas_return'] = MeasReturnType.SINGLE

        analysis_results, figures = super()._run_analysis(experiment_data)

        popt = next(res for res in analysis_results if res.name == PARAMS_ENTRY_PREFIX + self.name).value.params

        if self.options.plot:
            ahat_plotter = CurvePlotter(MplDrawer())
            ahat_plotter.set_figure_options(
                xlabel='Meas LO frequency offset',
                xval_unit='Hz',
                ylabel='Rabi amplitude'
            )

        ahats = []

        component_index = experiment_data.metadata['component_child_index']
        for idx, (analysis, child_index, offset) in enumerate(zip(self._analyses, component_index, self.freq_offsets)):
            result_name = PARAMS_ENTRY_PREFIX + analysis.name
            child_data = experiment_data.child_data(child_index)

            widths = np.array(list(d['metadata']['xval'] for d in child_data.data()))
            samples = unp.nominal_values(DataProcessor('memory')(child_data.data()))

            boundary, losses = fit_boundary(widths, samples, popt['freq'], popt['phase'],
                                            self.options.maxiter, self.options.convergence)

            analysis_results.append(AnalysisResultData(name=f'boundary_{idx}', value=boundary))

            title = f'{offset * 1.e-3:+.1f} kHz'

            if losses is not None:
                analysis_results.append(AnalysisResultData(name=f'loss_{idx}', value=losses))
                ## Loss curve
                figures.append(draw_loss(losses, title=title))
                ## Rabi amplitude
                ahats.append(-losses[-1])

            ## IQ plots
            figures.append(draw_iq_boundary(widths, samples, boundary, title=title))

        if self.options.plot and ahats:
            ahat_plotter.set_series_data(
                'ahat',
                x=self.freq_offsets,
                y=ahats
            )

            figures.append(ahat_plotter.figure())

        return analysis_results, figures
