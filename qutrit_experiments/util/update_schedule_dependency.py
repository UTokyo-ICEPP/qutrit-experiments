"""A fix for qiskit-experiments bug."""
from typing import Tuple, Union, Optional
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.calibration_utils import (
    validate_channels,
    reference_info,
    _get_node_index
)
from qiskit_experiments.calibration_management.calibration_key_types import ScheduleKey
from rustworkx import PyDiGraph  # pylint: disable=no-name-in-module


def update_schedule_dependency(schedule: ScheduleBlock, dag: PyDiGraph, key: ScheduleKey):
    """Update a DAG of schedule dependencies.

    Args:
        schedule: A ScheduleBlock that potentially has references to other schedules
            that are already present in the dag.
        dag: A directed acyclic graph that encodes schedule dependencies using references.
        key: The schedule key which also contains the qubits.
    """
    # First, check if the node is already in the DAG.
    try:
        # If it already is in the DAG we remove the existing edges and add the new ones later.
        parent_idx = dag.nodes().index(key)
        for successor in dag.successors(parent_idx):
            dag.remove_edge(parent_idx, dag.nodes().index(successor))

    except ValueError:
        # The schedule is not in the DAG: we add a new node.
        parent_idx = dag.add_node(key)

    for reference in schedule.references:
        #  BEGIN EDIT yiiyama
        # ref_key = ScheduleKey(reference[0], key.qubits)
        ref_schedule_name, ref_qubits = reference_info(reference, key.qubits)
        ref_key = ScheduleKey(ref_schedule_name, ref_qubits)
        #  END EDIT yiiyama
        dag.add_edge(parent_idx, _get_node_index(ref_key, dag), None)


def update_add_schedule(self):
    def add_schedule(
        schedule: ScheduleBlock,
        qubits: Union[int, Tuple[int, ...]] = None,
        num_qubits: Optional[int] = None,
    ):
        """Add a schedule block and register its parameters.

        Schedules that use Call instructions must register the called schedules separately.

        Args:
            schedule: The :class:`ScheduleBlock` to add.
            qubits: The qubits for which to add the schedules. If None or an empty tuple is
                given then this schedule is the default schedule for all qubits and, in this
                case, the number of qubits that this schedule act on must be given.
            num_qubits: The number of qubits that this schedule will act on when exported to
                a circuit instruction. This argument is optional as long as qubits is either
                not None or not an empty tuple (i.e. default schedule).

        Raises:
            CalibrationError: If schedule is not an instance of :class:`ScheduleBlock`.
            CalibrationError: If a schedule has assigned references.
            CalibrationError: If several parameters in the same schedule have the same name.
            CalibrationError: If the schedule name starts with the prefix of ScheduleBlock.
            CalibrationError: If the schedule calls subroutines that have not been registered.
            CalibrationError: If a :class:`Schedule` is Called instead of a :class:`ScheduleBlock`.
            CalibrationError: If a schedule with the same name exists and acts on a different
                number of qubits.

        """
        self._has_manually_added_schedule = True

        qubits = self._to_tuple(qubits)

        if len(qubits) == 0 and num_qubits is None:
            raise CalibrationError("Both qubits and num_qubits cannot simultaneously be None.")

        num_qubits = len(qubits) or num_qubits

        if not isinstance(schedule, ScheduleBlock):
            raise CalibrationError(f"{schedule.name} is not a ScheduleBlock.")

        if len(schedule.references) != len(schedule.references.unassigned()):
            raise CalibrationError(
                f"Cannot add {schedule} with assigned references. {self.__class__.__name__} only "
                "accepts schedules without references or schedules with references by name."
            )

        sched_key = ScheduleKey(schedule.name, qubits)

        # Ensure one to one mapping between name and number of qubits.
        if sched_key in self._schedules_qubits and self._schedules_qubits[sched_key] != num_qubits:
            raise CalibrationError(
                f"Cannot add schedule {schedule.name} acting on {num_qubits} qubits."
                "self already contains a schedule with the same name acting on "
                f"{self._schedules_qubits[sched_key]} qubits. Remove old schedule first."
            )

        # check that channels, if parameterized, have the proper name format.
        if schedule.name.startswith(ScheduleBlock.prefix):
            raise CalibrationError(
                f"{self.__class__.__name__} uses the `name` property of the schedule as part of a "
                f"database key. Using the automatically generated name {schedule.name} may have "
                f"unintended consequences. Please define a meaningful and unique schedule name."
            )

        param_indices = validate_channels(schedule)

        # Check that subroutines are present.
        for reference in schedule.references:
            #  BEGIN EDIT yiiyama
            # self.get_template(*reference_info(reference, qubits))
            ref_name, ref_qubits = reference_info(reference, qubits)
            if len(qubits) == 0:
                # ref_qubits is a tuple of "logical" qubits
                # First check if the reference is an unbound schedule
                if (nq := self._schedules_qubits.get(ScheduleKey(ref_name, ()))) is not None:
                    if nq != len(ref_qubits):
                        raise CalibrationError(f'Reference with {len(ref_qubits)} made to template'
                                               f' {ref_name} on {nq} qubits')
                else:
                    # If not found, look for at least one bound schedule with the right num_qubits
                    has_schedule = False
                    for key, nq in self._schedules_qubits.items():
                        if key.schedule == ref_name:
                            has_schedule = True
                            if len(key.qubits) == len(ref_qubits):
                                break
                    else:
                        msg = f'Could not find schedule {ref_name}'
                        if has_schedule:
                            msg += f' with {len(ref_qubits)} qubits'
                        raise CalibrationError(msg)
            else:
                self.get_template(ref_name, ref_qubits)
            #  END EDIT yiiyama

        # Clean the parameter to schedule mapping. This is needed if we overwrite a schedule.
        self._clean_parameter_map(schedule.name, qubits)

        # Add the schedule.
        self._schedules[sched_key] = schedule
        self._schedules_qubits[sched_key] = num_qubits

        # Update the schedule dependency.
        update_schedule_dependency(schedule, self._schedule_dependency, sched_key)

        # Register parameters that are not indices.
        params_to_register = set()
        for param in schedule.parameters:
            if param not in param_indices:
                params_to_register.add(param)

        if len(params_to_register) != len(set(param.name for param in params_to_register)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        for param in params_to_register:
            self._register_parameter(param, qubits, schedule)

    self.add_schedule = add_schedule
