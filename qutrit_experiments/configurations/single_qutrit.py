from .qutrit import (qutrit_rough_frequency, qutrit_rough_amplitude, qutrit_semifine_frequency,
                     qutrit_fine_frequency)
from ..experiment_config import register_exp


def register_single_qutrit_exp(function):
    def gen(runner):
        exp_config = function(runner)
        exp_config.physical_qubits = [runner.program_data['qubit']]
        return exp_config

    register_exp(gen, exp_type=function.__name__)

register_single_qutrit_exp(qutrit_rough_frequency)
