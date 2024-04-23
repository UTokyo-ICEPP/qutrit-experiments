from functools import wraps
import logging
import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability

from ..data_processing import ReadoutMitigation
from ..experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def add_readout_mitigation(gen=None, *, logical_qubits=None, expval=False):
    """Decorator to add a readout error mitigation node to the DataProcessor."""
    if gen is None:
        def wrapper(gen):
            return add_readout_mitigation(gen, logical_qubits=logical_qubits, expval=expval)
        return wrapper

    @wraps(gen)
    def converted_gen(runner, *args, **kwargs):
        config = gen(runner, *args, **kwargs)
        configure_readout_mitigation(runner, config, logical_qubits=logical_qubits, expval=expval)
        return config

    return converted_gen

def configure_readout_mitigation(runner, config, logical_qubits=None, expval=False):
    if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
        logger.warning('MeasLevel is not CLASSIFIED; no readout mitigation. run_options=%s',
                       config.run_options)
        return

    if logical_qubits is not None:
        qubits = tuple(config.physical_qubits[q] for q in logical_qubits)
    else:
        qubits = tuple(config.physical_qubits)

    for mitigator_qubits, mitigator in runner.program_data.get('readout_mitigator', {}).items():
        if set(qubits) < set(mitigator_qubits):
            break
    else:
        logger.warning('Correlated readout mitigator for qubits %s not found.', qubits)
        return

    # CorrelatedReadoutMitigator.assignment_matrix() implicitly sorts the qubits through the use of
    # set, so we reorder the axes. Also note that unused qubits are assumed to be at |0> state
    matrix = mitigator.assignment_matrix(qubits)
    indices = list(reversed([mitigator_qubits.index(iq) for iq in qubits]))
    sorted_indices = list(reversed(sorted(indices)))
    if indices != sorted_indices:
        nq = len(indices)
        transpose = [sorted_indices.index(i) for i in indices] # row reordering
        transpose += [t + nq for t in transpose] # column reordering
        matrix = matrix.reshape((2,) * (2 * nq)).transpose(transpose).reshape((2 ** nq,) * 2)

    if (processor := config.analysis_options.get('data_processor')) is None:
        nodes = [
            ReadoutMitigation(matrix),
            Probability(config.analysis_options.get('outcome', '1' * len(qubits)))
        ]
        if expval:
            nodes.append(BasisExpectationValue())
        config.analysis_options['data_processor'] = DataProcessor('counts', nodes)
    else:
        try:
            insert_pos = next(inode for inode, node in enumerate(processor._nodes)
                              if isinstance(node, Probability))
        except StopIteration:
            insert_pos = 0
        processor._nodes.insert(insert_pos, ReadoutMitigation(matrix))

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
