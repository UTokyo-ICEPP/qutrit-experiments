# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Single qutrit calibration and characterization experiments."""
from functools import wraps
from .qutrit import (
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_discriminator,
    qutrit_discriminator_post,
    qubit_assignment_error,
    qubit_assignment_error_post,
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


def register_single_qutrit_exp(function):
    @wraps(function)
    def conf_gen(runner):
        return function(runner, runner.program_data['qutrit'])

    register_exp(conf_gen)


qutrit_functions = [
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_discriminator,
    qubit_assignment_error,
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
    register_single_qutrit_exp(func)

register_post(qutrit_discriminator_post, exp_type='qutrit_discriminator')
register_post(qubit_assignment_error_post, exp_type='qubit_assignment_error')
register_post(qutrit_assignment_error_post, exp_type='qutrit_assignment_error')
