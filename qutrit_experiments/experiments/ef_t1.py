"""A T1 experiment for qutrits.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import lmfit
from scipy.integrate import odeint
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, CircuitInstruction
from qiskit.providers import Backend
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, ExperimentData, Options
from qiskit_experiments.library import T1
from qiskit_experiments.visualization import CurvePlotter, MplDrawer
import qiskit_experiments.curve_analysis as curve

from ..common.ef_space import EFSpaceExperiment
from ..common.gates import X12Gate
from ..common.readout_mitigation import MCMLocalReadoutMitigation

class EFT1(EFSpaceExperiment, T1):
    _initial_xgate = False

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delays = np.linspace(0., 5.e-4, 30)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Optional[Union[List[float], np.array]] = None,
        backend: Optional[Backend] = None
    ):
        if (delays_exp := delays) is None:
            delays_exp = self._default_experiment_options().delays

        super().__init__(physical_qubits, delays_exp, backend=backend)
        self.analysis = EFT1Analysis()

        if delays is not None:
            self.set_experiment_options(delays=delays)

    def circuits(self) -> List[QuantumCircuit]:
        circuits = super().circuits()
        for circuit in circuits:
            for idx, inst in enumerate(circuit.data):
                if inst.operation.name == 'x':
                    circuit.data.insert(idx + 1,
                                        CircuitInstruction(X12Gate(), [self.physical_qubits[0]]))
                    break

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        circuits = super()._transpiled_circuits()
        for circuit in circuits:
            creg = ClassicalRegister(size=2)
            circuit.remove_final_measurements(inplace=True)
            circuit.add_register(creg)
            # Post-transpilation - logical qubit = physical qubit
            circuit.measure(self.physical_qubits[0], creg[0])
            circuit.x(self.physical_qubits[0])
            circuit.measure(self.physical_qubits[0], creg[1])

        return circuits


class EFT1Analysis(BaseAnalysis):
    """Run fit with rate equations."""

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()

        plotter = CurvePlotter(MplDrawer())
        plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Probability",
            xval_unit="s",
        )
        options.update_options(
            plotter=plotter,
            assignment_matrix=None,
            alpha_prior=[0.5, 0.5],
        )
        return options

    def _run_data_processing(
        self,
        raw_data: List[Dict],
    ) -> curve.CurveData:

        xdata = []
        ydata = []
        y_err = []
        shots = []
        data_allocation = []
        for datum in raw_data:
            if self.options.assignment_matrix is not None:
                raw_count = {
                    "0": datum["counts"].get("10", 0),
                    "1": datum["counts"].get("01", 0),
                    "2": datum["counts"].get("11", 0),
                    "3": datum["counts"].get("00", 0),
                }
                count_sum = sum(raw_count.values())
                mitigator = MCMLocalReadoutMitigation(self.options.assignment_matrix)
                mit_prob = mitigator.quasi_probabilities(raw_count)
                mit_prob = mit_prob.nearest_probability_distribution()
                trit_count = {str(k): int(v * count_sum) for k, v in mit_prob.items()}
            else:
                trit_count = {
                    "0": datum["counts"].get("10", 0),
                    "1": datum["counts"].get("01", 0),
                    "2": datum["counts"].get("11", 0),
                }
            net_shots = sum(trit_count.values())

            for i, state in enumerate(("0", "1", "2")):
                freq = trit_count.get(state, 0)
                alpha_posterior = [
                    freq + self.options.alpha_prior[0],
                    net_shots - freq + self.options.alpha_prior[1],
                ]
                alpha_sum = sum(alpha_posterior)
                p_mean = alpha_posterior[0] / alpha_sum
                p_var = p_mean * (1 - p_mean) / (alpha_sum + 1)

                xdata.append(datum["metadata"]["xval"])
                ydata.append(p_mean)
                y_err.append(p_var)
                shots.append(net_shots)
                data_allocation.append(i)

        return curve.CurveData(
            x=np.asarray(xdata, dtype=float),
            y=np.asarray(ydata, dtype=float),
            y_err=np.asarray(y_err, dtype=float),
            shots=np.asarray(shots, dtype=int),
            data_allocation=np.asarray(data_allocation, dtype=int),
            labels=["0", "1", "2"],
        )

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:

        analysis_results = []
        figures = []

        curve_data = self._run_data_processing(experiment_data.data())
        p0_exp = curve_data.get_subset_of("0")
        p1_exp = curve_data.get_subset_of("1")
        p2_exp = curve_data.get_subset_of("2")

        x = p0_exp.x
        assert np.array_equal(x, p1_exp.x)
        assert np.array_equal(x, p2_exp.x)

        for data in p0_exp, p1_exp, p2_exp:
            self.options.plotter.set_series_data(
                data.labels[0],
                x_formatted=data.x,
                y_formatted=data.y,
                y_formatted_err=data.y_err,
            )

        def _rate_equation(p, _, g10, g20, g21):
            dp0dt = g10 * p[1] + g20 * p[2]
            dp1dt = -g10 * p[1] + g21 * p[2]
            dp2dt = -(g20 + g21) * p[2]

            return [dp0dt, dp1dt, dp2dt]

        def _objective(_params):
            p_ini = [_params["pi0"].value, _params["pi1"].value, _params["pi2"].value]
            args = [_params[k].value for k in ["g10", "g20", "g21"]]
            p = odeint(_rate_equation, p_ini, x, args=tuple(args))
            resids = [
                p[:, 0] - p0_exp.y,
                p[:, 1] - p1_exp.y,
                p[:, 2] - p2_exp.y,
            ]
            return np.concatenate(resids)

        # initial guess
        p1_dominant = p2_exp.y < 0.2
        params = lmfit.Parameters()
        params.add(
            name="pi0",
            value=0.0,
            min=0.0,
            max=1.0,
            vary=True,
        )
        params.add(
            name="pi1",
            value=0.0,
            min=0.0,
            max=1.0,
            vary=True,
        )
        params.add(
            name="pi2",
            value=1.0,
            min=0.0,
            max=1.0,
            vary=True,
        )
        params.add(
            name="g10",
            value=- curve.guess.exp_decay(p1_exp.x[p1_dominant], p1_exp.y[p1_dominant]),
            min=0.0,
            vary=True,
        )
        params.add(
            name="g20",
            value=0.0,
            min=0.0,
            vary=True,
        )
        params.add(
            name="g21",
            value=- curve.guess.exp_decay(p2_exp.x, p2_exp.y),
            min=0.0,
            vary=True,
        )
        try:
            res = lmfit.minimize(
                fcn=_objective,
                params=params,
                method="least_squares",
                scale_covar=False,
                nan_policy="omit",
            )
            fit_data = curve.utils.convert_lmfit_result(
                res, [], curve_data.x, curve_data.y
            )
            analysis_results.extend(
                [
                    AnalysisResultData(
                        name="@Parameters_TritT1RateAnalysis",
                        value=fit_data,
                    ),
                    AnalysisResultData(
                        name="Gamma10",
                        value=fit_data.ufloat_params["g10"],
                        chisq=fit_data.reduced_chisq,
                    ),
                    AnalysisResultData(
                        name="Gamma20",
                        value=fit_data.ufloat_params["g20"],
                        chisq=fit_data.reduced_chisq,
                    ),
                    AnalysisResultData(
                        name="Gamma21",
                        value=fit_data.ufloat_params["g21"],
                        chisq=fit_data.reduced_chisq,
                    ),
                ]
            )
            self.options.plotter.set_supplementary_data(primary_results=analysis_results[1:])

            # Create fit lines
            p_ini_fit = [
                fit_data.ufloat_params["pi0"].n,
                fit_data.ufloat_params["pi1"].n,
                fit_data.ufloat_params["pi2"].n,
            ]
            args_fit = [
                fit_data.ufloat_params["g10"].n,
                fit_data.ufloat_params["g20"].n,
                fit_data.ufloat_params["g21"].n,
            ]
            x_interp = np.linspace(x[0], x[-1], 100)
            p_fit = odeint(_rate_equation, p_ini_fit, x_interp, args=tuple(args_fit))
            self.options.plotter.set_series_data(
                "0",
                x_interp=x_interp,
                y_interp=p_fit[:, 0],
            )
            self.options.plotter.set_series_data(
                "1",
                x_interp=x_interp,
                y_interp=p_fit[:, 1],
            )
            self.options.plotter.set_series_data(
                "2",
                x_interp=x_interp,
                y_interp=p_fit[:, 2],
            )
            analysis_results.append(
                AnalysisResultData(
                    name="@DataFit_EFT1Analysis",
                    value={
                        "0": {
                            "x": x_interp,
                            "y": p_fit[:, 0],
                        },
                        "1": {
                            "x": x_interp,
                            "y": p_fit[:, 1],
                        },
                        "2": {
                            "x": x_interp,
                            "y": p_fit[:, 2],
                        },
                    }
                )
            )
        except Exception:
            pass

        # Save data points
        raw_data = {}
        for data in p0_exp, p1_exp, p2_exp:
            raw_data[data.labels[0]] = data
        analysis_results.append(
            AnalysisResultData(
                name="@Data_EFT1Analysis",
                value=raw_data,
            )
        )
        figures.append(self.options.plotter.figure())
        return analysis_results, figures
