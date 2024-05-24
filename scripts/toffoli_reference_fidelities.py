#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

if __name__ == '__main__':
    from qutrit_experiments.script_util.program_config import get_program_config
    program_config = get_program_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit import QuantumCircuit, transpile, schedule
    from qiskit_experiments.library.characterization import CorrelatedReadoutError
    from qiskit_experiments.library.tomography import ProcessTomography
    from qiskit_experiments.data_processing import DataProcessor
    from qutrit_experiments.data_processing import ReadoutMitigation
    from qutrit_experiments.experiments.process_tomography import CircuitTomography, CircuitTomographyAnalysis
    from qutrit_experiments.experiment_config import ExperimentConfig, ParallelExperimentConfig
    from qutrit_experiments.programs.common import (load_calibrations, setup_backend,
                                                    setup_data_dir, setup_runner)
    from qutrit_experiments.runners.experiments_runner import ExperimentsRunner
    from qutrit_experiments.configurations.common import configure_qpt_readout_mitigation

    def qubits_suffix(qubits):
        return '_'.join(map(str, qubits))

    # Create the data directory
    setup_data_dir(program_config)
    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    # Instantiate the backend
    backend = setup_backend(program_config)
    qubits_list = [tuple(program_config['qubits'][i:i + 3])
                   for i in range(0, len(program_config['qubits']), 3)]

    runner = setup_runner(backend, program_config)
    runner.job_retry_interval = 120

    ## Readout mitigation

    config = ParallelExperimentConfig(
        [
            ExperimentConfig(
                CorrelatedReadoutError,
                qubits,
                exp_type=f'readout_error_{qubits_suffix(qubits)}'
            ) for qubits in qubits_list
        ],
        run_options={'shots': 10000},
        exp_type='readout_error'
    )
    data = runner.run_experiment(config)

    runner.program_data['readout_mitigator'] = {}
    for child_data in data.child_data():
        mitigator = child_data.analysis_results('Correlated Readout Mitigator').value
        runner.program_data['readout_mitigator'][tuple(child_data.metadata['physical_qubits'])] = mitigator

    ## Default decomposition

    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)

    config = ParallelExperimentConfig(
        run_options={'shots': 2000},
        exp_type='qpt_opt3'
    )

    for qubits in qubits_list:
        tcirc = transpile(circuit, backend=backend, initial_layout=qubits, optimization_level=3)
        lcirc = QuantumCircuit(3)
        qmap = {q: lcirc.qregs[0][iq] for iq, q in enumerate(qubits)}
        for inst, qargs, _ in tcirc.data:
            mapped_qargs = [qmap[tcirc.find_bit(i).index] for i in qargs]
            lcirc._append(inst, mapped_qargs, [])

        subconfig = ExperimentConfig(
            CircuitTomography,
            qubits,
            args={'circuit': lcirc, 'target_circuit': circuit},
            exp_type=f'qpt_opt3_{qubits_suffix(qubits)}'
        )
        configure_qpt_readout_mitigation(runner, subconfig)
        config.subexperiments.append(subconfig)
    data_default = runner.run_experiment(config, block_for_results=False)

    ## 8CX decomposition

    circuit = QuantumCircuit(3)
    circuit.t(0)
    circuit.t(1)
    circuit.h(2)
    circuit.cx(0, 1)
    circuit.t(2)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.t(2)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.tdg(2)
    circuit.tdg(1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.tdg(2)
    circuit.cx(1, 2)
    circuit.h(2)

    target_circuit = QuantumCircuit(3)
    target_circuit.ccx(0, 1, 2)

    config = ParallelExperimentConfig(
        run_options={'shots': 2000},
        exp_type='qpt_8cx'
    )

    for qubits in qubits_list:
        tcirc = transpile(circuit, backend=backend, initial_layout=qubits)
        lcirc = QuantumCircuit(3)
        qmap = {q: lcirc.qregs[0][iq] for iq, q in enumerate(qubits)}
        for inst, qargs, _ in tcirc.data:
            mapped_qargs = [qmap[tcirc.find_bit(i).index] for i in qargs]
            lcirc._append(inst, mapped_qargs, [])

        subconfig = ExperimentConfig(
            CircuitTomography,
            qubits,
            args={'circuit': lcirc, 'target_circuit': circuit},
            exp_type=f'qpt_8cx_{qubits_suffix(qubits)}'
        )
        configure_qpt_readout_mitigation(runner, subconfig)
        config.subexperiments.append(subconfig)

    data_8cx = runner.run_experiment(config, block_for_results=False)

    data_default.block_for_results()
    data_8cx.block_for_results()

    print('Default decomposition:', [child_data.analysis_results('process_fidelity').value for child_data in data_default.child_data()])
    print('8CX decomposition:', [child_data.analysis_results('process_fidelity').value for child_data in data_8cx.child_data()])
