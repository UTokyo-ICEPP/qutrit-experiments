# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Common experiment configurations and functions for setting up experiment configurations."""

from collections.abc import Sequence
from functools import wraps
import logging
from typing import Optional, Union
from qiskit.qobj.utils import MeasLevel
from qiskit.result import LocalReadoutMitigator
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework import ExperimentData

from ..data_processing import ReadoutMitigation
from ..experiment_config import ExperimentConfig
from ..runners import ExperimentsRunner

LOG = logging.getLogger(__name__)


def add_readout_mitigation(gen=None, *, logical_qubits=None, probability=True, expval=False):
    """Decorator to add a ReadoutMitigation node to the DataProcessor."""
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


def configure_readout_mitigation(
    runner: ExperimentsRunner,
    config: ExperimentConfig,
    logical_qubits: Optional[Sequence[int]] = None,
    probability: bool = True,
    expval: bool = False
) -> None:
    """Set data_processor with readout mitigation in the analysis_options of the config.

    This function merely sets up the DataProcessor in the ``config.analysis_options`` dict and does
    not e.g. check whether the actual analysis class that will be instantiated utilizes the DP. If
    there is already a data_processor entry in analysis_options, a ReadoutMitigation node is
    inserted at position 0 of the DP nodes list. Otherwise a new DataProcessor object is created
    with ``'counts'`` as the data key. Outcome of the measurement to process is determined by the
    ``'outcome'`` option in **analysis_options of the config** (not in the actual Analysis code),
    which if not found will be set to ``'1' * nq`` where ``nq`` is the number of logical_qubits if
    it is set, otherwise the number of physical qubits of the configuration.

    Args:
        runner: ExperimentsRunner object.
        config: ExperimentConfig to configure the readout error mitigation on.
        logical_qubits: Indices of ``config.physical_qubits`` where readout mitigation should be
            configured on.
        probability: Whether to append the Probability node to the newly created DataProcessor.
        expval: Whether to append the BasisExpectationValue node to the newly created DataProcessor.
    """
    if config.run_options.get('meas_level', MeasLevel.CLASSIFIED) != MeasLevel.CLASSIFIED:
        LOG.warning('MeasLevel is not CLASSIFIED; no readout mitigation. run_options=%s',
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
        LOG.warning('Correlated readout mitigator for qubits %s not found.', qubits)
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
    """Decorator to set the ``readout_mitigator`` analysis option of a TomographyExperiment."""
    @wraps(gen)
    def converted_gen(runner):
        config = gen(runner)
        configure_qpt_readout_mitigation(runner, config)
        return config

    return converted_gen


def configure_qpt_readout_mitigation(
    runner: ExperimentsRunner,
    config: ExperimentConfig
) -> None:
    """Set the ``readout_mitigator`` analysis option to a TomographyExperiment configuration.

    Args:
        runner: ExperimentsRunner object.
        config: ExperimentConfig to set the readout_mitigator option on ``analysis_options``.
    """
    for mitigator_qubits, mitigator in runner.program_data.get('readout_mitigator', {}).items():
        if set(config.physical_qubits) <= set(mitigator_qubits):
            matrices = [mitigator.assignment_matrix((qubit,)) for qubit in config.physical_qubits]
            local_mitigator = LocalReadoutMitigator(matrices, config.physical_qubits)
            config.analysis_options['readout_mitigator'] = local_mitigator
            return

    LOG.warning('Correlated readout mitigator for qubits %s not found.', config.physical_qubits)


def qubits_assignment_error(
    runner: ExperimentsRunner,
    qubits: Union[int, Sequence[int]]
) -> ExperimentConfig:
    """Return an ExperimentConfig for CorrelatedReadoutError.

    Args:
        runner: ExperimentsRunner object.
        qubits: Physical qubit(s) of the configuration.
    """
    from ..experiments.readout_error import CorrelatedReadoutError
    if isinstance(qubits, int):
        qubits = [qubits]
    return ExperimentConfig(
        CorrelatedReadoutError,
        qubits
    )


def qubits_assignment_error_post(
    runner: ExperimentConfig,
    experiment_data: ExperimentData
) -> None:
    """Postexperiment for qubits_assignment_error."""
    mitigator = experiment_data.analysis_results('Correlated Readout Mitigator', block=False).value
    physical_qubits = tuple(experiment_data.metadata['physical_qubits'])
    runner.program_data.setdefault('readout_mitigator', {})[physical_qubits] = mitigator
