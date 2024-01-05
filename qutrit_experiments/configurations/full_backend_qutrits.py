# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Single qutrit calibration and characterization experiments for a full backend."""
from .qutrit import (qutrit_rough_frequency, qutrit_rough_amplitude,
                     qutrit_semifine_frequency,
                     qutrit_fine_frequency, qutrit_rough_x_drag, qutrit_rough_sx_drag,
                     qutrit_fine_sx_amplitude, qutrit_fine_sx_drag, qutrit_fine_x_amplitude,
                     qutrit_fine_x_drag, qutrit_x12_stark_shift, qutrit_x_stark_shift,
                     qutrit_sx_stark_shift, qutrit_rotary_stark_shift, qutrit_assignment_error,
                     qutrit_assignment_error_post, qutrit_t1)
from ..experiment_config import (BatchExperimentConfig, ExperimentConfig, ParallelExperimentConfig,
                                 register_exp, register_post)


def register_backend_qutrit_exp(function):
    def conf_gen(runner):
        parallel_conf = ParallelExperimentConfig(
            experiment_options={'max_circuits': 50}
        )
        for igroup, qubit_group in enumerate(runner.qubit_grouping):
            batch_conf = BatchExperimentConfig()
            for qubit in qubit_group:
                qubit_config = function(runner, qubit)
                qubit_config.exp_type = f'{function.__name__}-q{qubit}'
                if not batch_conf.subexperiments:
                    batch_conf.exp_type = f'{function.__name__}-g{igroup}'
                    batch_conf.analysis = qubit_config.analysis
                    default_run_options = dict(
                        qubit_config.cls._default_run_options().items()
                    )
                    # Run options on intermediate BatchExperiments have no effect but warnings may
                    # be issued if meas_level and meas_return_type are different between the
                    # batched experiment and any container experiments, so we set the run options
                    # here too
                    batch_conf.run_options.update(default_run_options)
                    batch_conf.run_options.update(qubit_config.run_options)
                    if not parallel_conf.subexperiments:
                        parallel_conf.analysis = qubit_config.analysis
                        parallel_conf.run_options.update(default_run_options)
                        parallel_conf.run_options.update(qubit_config.run_options)
                batch_conf.subexperiments.append(qubit_config)
            parallel_conf.subexperiments.append(batch_conf)

        return parallel_conf

    register_exp(conf_gen, exp_type=function.__name__)

def register_backend_qutrit_postexp(function):
    def postexp(runner, experiment_data):
        for parallel_data in experiment_data.child_data():
            for qutrit_data in parallel_data.child_data():
                function(runner, qutrit_data)

    register_post(postexp, exp_type=function.__name__[:-5])


qutrit_functions = [
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift,
    qutrit_assignment_error,
    qutrit_t1
]
for func in qutrit_functions:
    register_backend_qutrit_exp(func)

register_backend_qutrit_postexp(qutrit_assignment_error_post)
