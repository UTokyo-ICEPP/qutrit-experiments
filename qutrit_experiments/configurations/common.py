from functools import wraps
import logging
import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit.result import LocalReadoutMitigator
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability

from ..data_processing import ReadoutMitigation
from ..experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def add_readout_mitigation(gen=None, *, logical_qubits=None, probability=True, expval=False):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    if gen is None:
        def wrapper(gen):
            return add_readout_mitigation(gen, logical_qubits=logical_qubits,
                                          probability=probability, expval=expval)
        return wrapper

    @wraps(gen)
    def converted_gen(runner, *args, **kwargs):
        config = gen(runner, *args, **kwargs)
        configure_readout_mitigation(runner, config, logical_qubits=logical_qubits,
                                     probability=probability, expval=expval)
        return config

    return converted_gen

def configure_readout_mitigation(runner, config, logical_qubits=None, probability=True,
                                 expval=False):
    if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
        logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation. run_options=%s',
                       config.run_options)
        return

    if logical_qubits is not None:
        qubits = tuple(config.physical_qubits[q] for q in logical_qubits)
    else:
        qubits = tuple(config.physical_qubits)

    for mitigator_qubits, mitigator in runner.program_data.get('readout_mitigator', {}).items():
        if set(qubits) <= set(mitigator_qubits):
            break
    else:
        logger.warning('Correlated readout mitigator for qubits %s not found.', qubits)
        return

    mit_node = ReadoutMitigation(readout_mitigator=mitigator, physical_qubits=qubits)

    if (processor := config.analysis_options.get('data_processor')) is None:
        nodes = [mit_node]
        if probability:
            nodes.append(Probability(config.analysis_options.get('outcome', '1' * len(qubits))))
        if expval:
            nodes.append(BasisExpectationValue())
        config.analysis_options['data_processor'] = DataProcessor('counts', nodes)
    else:
        processor._nodes.insert(0, mit_node)

def add_qpt_readout_mitigation(gen):
    @wraps(gen)
    def converted_gen(runner):
        config = gen(runner)
        configure_qpt_readout_mitigation(runner, config)
        return config

    return converted_gen

def configure_qpt_readout_mitigation(runner, config):
    for mitigator_qubits, mitigator in runner.program_data.get('readout_mitigator', {}).items():
        if set(config.physical_qubits) <= set(mitigator_qubits):
            matrices = [mitigator.assignment_matrix((qubit,)) for qubit in config.physical_qubits]
            local_mitigator = LocalReadoutMitigator(matrices, config.physical_qubits)
            config.analysis_options['readout_mitigator'] = local_mitigator
            return

    logger.warning('Correlated readout mitigator for qubits %s not found.', config.physical_qubits)

def qubits_assignment_error(runner, qubits):
    """Template configuration generator for CorrelatedReadoutError."""
    from ..experiments.readout_error import CorrelatedReadoutError
    if isinstance(qubits, int):
        qubits = [qubits]
    return ExperimentConfig(
        CorrelatedReadoutError,
        qubits
    )

def qubits_assignment_error_post(runner, experiment_data):
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator', block=False).value
    physical_qubits = tuple(experiment_data.metadata['physical_qubits'])
    runner.program_data.setdefault('readout_mitigator', {})[physical_qubits] = mitigator
