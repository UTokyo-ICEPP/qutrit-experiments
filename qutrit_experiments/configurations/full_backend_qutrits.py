# pylint: disable=function-redefined, unused-argument
"""Single qutrit calibration and characterization experiments for a full backend."""
from functools import wraps
import logging
from .common import qubits_assignment_error, qubits_assignment_error_post
from .qutrit import (
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
    qutrit_sx12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift,
    qutrit_assignment_error,
    qutrit_assignment_error_post,
    qutrit_t1,
    qutrit_x12_irb
)
from ..experiment_config import register_exp, register_post

logger = logging.getLogger(__name__)


def register_backend_qutrit_exp(function):
    @wraps(function)
    def conf_gen(runner):
        return runner.make_batch_config(function, exp_type=function.__name__)
    register_exp(conf_gen)


def register_backend_qutrit_postexp(function):
    @wraps(function)
    def postexp(runner, experiment_data):
        for qubit, qutrit_data in runner.decompose_data(experiment_data).items():
            try:
                function(runner, qutrit_data)
            except Exception as ex:
                logger.error('Postexperiment error at qubit %d: %s', qubit, ex)

    register_post(postexp, exp_type=function.__name__[:-5])


qutrit_functions = [
    qutrit_rough_frequency,
    #qutrit_rough_amplitude,
    qubits_assignment_error,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_sx12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift,
    qutrit_assignment_error,
    qutrit_t1,
    qutrit_x12_irb
]
for func in qutrit_functions:
    register_backend_qutrit_exp(func)

def qutrit_rough_amplitude_parallel(runner):
    qubit_grouping = runner.get_qubit_grouping(active_qubits=runner.qubits, max_group_size=5)
    conf = runner.make_batch_config(qutrit_rough_amplitude, exp_type='qutrit_rough_amplitude',
                                    qubit_grouping=qubit_grouping)
    conf.experiment_options['max_circuits'] = 150
    return conf
register_exp(qutrit_rough_amplitude_parallel, exp_type='qutrit_rough_amplitude')

register_backend_qutrit_postexp(qubits_assignment_error_post)
register_backend_qutrit_postexp(qutrit_assignment_error_post)
