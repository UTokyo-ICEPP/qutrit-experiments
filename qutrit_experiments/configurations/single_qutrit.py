"""Single qutrit calibration and characterization experiments."""
import numpy as np

from .qutrit import (qutrit_rough_frequency, qutrit_rough_frequency_post, qutrit_rough_amplitude,
                     qutrit_semifine_frequency, qutrit_fine_frequency, qutrit_rough_x_drag,
                     qutrit_rough_sx_drag, qutrit_fine_sx_amplitude, qutrit_fine_sx_drag,
                     qutrit_fine_x_amplitude, qutrit_fine_x_drag, qutrit_x12_stark_shift,
                     qutrit_x_stark_shift, qutrit_sx_stark_shift, qutrit_rotary_stark_shift,
                     qutrit_assignment_error, qutrit_assignment_error_post, qutrit_t1)
from ..experiment_config import register_exp, register_post
from ..util.iq_classification import SingleQutritLinearIQClassifier


def register_single_qutrit_exp(function):
    def conf_gen(runner):
        return function(runner, runner.program_data['qubit'])

    register_exp(conf_gen, exp_type=function.__name__)

def register_single_qutrit_postexp(function):
    def postexp(runner, experiment_data):
        return function(runner, experiment_data, runner.program_data['qubit'])

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
    register_single_qutrit_exp(func)

register_single_qutrit_postexp(qutrit_rough_frequency_post)
register_single_qutrit_postexp(qutrit_assignment_error_post)

@register_post
def qutrit_rough_amplitude(runner, experiment_data):
    from ..util.ef_discriminator import ef_discriminator_analysis

    amps = np.array([d['metadata']['xval'] for d in experiment_data.data()])
    theta, dist = ef_discriminator_analysis(experiment_data, np.argmin(np.abs(amps)))
    runner.program_data['iq_classifier'] = {runner.program_data['qubit']:
                                            SingleQutritLinearIQClassifier(theta, dist)}
