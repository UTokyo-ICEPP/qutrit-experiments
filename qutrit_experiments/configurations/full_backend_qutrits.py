# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Single qutrit calibration and characterization experiments for a full backend."""
from .qutrit import (qutrit_rough_frequency, qutrit_rough_amplitude, qutrit_rough_amplitude_post,
                     qutrit_semifine_frequency,
                     qutrit_fine_frequency, qutrit_rough_x_drag, qutrit_rough_sx_drag,
                     qutrit_fine_sx_amplitude, qutrit_fine_sx_drag, qutrit_fine_x_amplitude,
                     qutrit_fine_x_drag, qutrit_x12_stark_shift, qutrit_x_stark_shift,
                     qutrit_sx_stark_shift, qutrit_rotary_stark_shift, qutrit_assignment_error,
                     qutrit_assignment_error_post, qutrit_t1)
from ..experiment_config import ExperimentConfig, register_exp, register_post


def register_backend_qutrit_exp(function):
    def conf_gen(runner):
        config = ExperimentConfig(subexperiments=[])
        for qubit in runner.active_qubits:
            subconfig = function(runner, qubit)
            subconfig.exp_type = f'{function.__name__}-q{qubit}'
            config.subexperiments.append(subconfig)
        return config

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

register_backend_qutrit_postexp(qutrit_rough_amplitude_post)
register_backend_qutrit_postexp(qutrit_assignment_error_post)
