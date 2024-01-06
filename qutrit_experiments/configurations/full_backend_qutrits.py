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
        return runner.make_batch_config(function, exp_type=function.__name__)
    register_exp(conf_gen, exp_type=function.__name__)


def register_backend_qutrit_postexp(function):
    def postexp(runner, experiment_data):
        for parallel_data in experiment_data.child_data():
            for qutrit_data in parallel_data.child_data():
                function(runner, qutrit_data)

    register_post(postexp, exp_type=function.__name__[:-5])


qutrit_functions = [
    qutrit_rough_frequency,
    #qutrit_rough_amplitude,
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

def qutrit_rough_amplitude_parallel(runner):
    active_qubits = runner.active_qubits
    current_max_size = max(len(group) for group in runner.qubit_grouping)
    runner.set_qubit_grouping(active_qubits=active_qubits, max_group_size=5)
    conf = runner.make_batch_config(qutrit_rough_amplitude, exp_type='qutrit_rough_amplitude')
    conf.experiment_options['max_circuits'] = 150
    runner.set_qubit_grouping(active_qubits=active_qubits, max_group_size=current_max_size)
    return conf
register_exp(qutrit_rough_amplitude_parallel, exp_type='qutrit_rough_amplitude')

register_backend_qutrit_postexp(qutrit_assignment_error_post)
