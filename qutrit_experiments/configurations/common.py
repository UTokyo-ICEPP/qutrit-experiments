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

    if (matrix := runner.program_data.get('readout_assignment_matrices', {}).get(qubits)) is None:
        logger.warning('Assignment matrix missing; no readout mitigation. qubits=%s', qubits)
        return

    if (processor := config.analysis_options.get('data_processor')) is None:
        nodes = [
            ReadoutMitigation(matrix),
            Probability(config.analysis_options.get('outcome', '1' * len(qubits)))
        ]
        if expval:
            nodes.append(BasisExpectationValue())
        config.analysis_options['data_processor'] = DataProcessor('counts', nodes)
    else:
        processor._nodes.insert(0, ReadoutMitigation(matrix))

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
    qubits = tuple(experiment_data.metadata['physical_qubits'])
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator', block=False).value
    prog_data = runner.program_data.setdefault('readout_assignment_matrices', {})
    for combination in [qubits[0:1], qubits[1:2], qubits[2:3], qubits[:2], qubits[1:3], qubits]:
        prog_data[combination] = mitigator.assignment_matrix(combination)
