from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass


class AddDDCalibration(TransformationPass):
    def __init__(self):
        super().__init__()
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for placement in ['left', 'right']:
            name = f'dd_{placement}'
            for node in dag.named_nodes(name):
                qubit = dag.find_bit(node.qargs[0]).index
                duration = node.op.params[0]
                if dag.calibrations.get(name, {}).get(((qubit,), (duration,))) is None:
                    sched = self.calibrations.get_schedule(name, qubit,
                                                           assign_params={'duration': duration})
                    dag.add_calibration(name, [qubit], sched, params=[duration])

        return dag
