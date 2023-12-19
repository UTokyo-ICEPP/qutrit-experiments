"""CompositeExperiment with an automatic child data structure setter."""

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.composite.composite_experiment import (CompositeExperiment
                                                                         as CompositeExperimentOrig)

from ..framework.postprocessed_experiment_data import PostprocessedExperimentData
from .composite_analysis import CompositeAnalysis


class CompositeExperiment(CompositeExperimentOrig):
    """CompositeExperiment with an automatic child data structure setter."""
    def _initialize_experiment_data(self) -> ExperimentData:
        """Initialize the return data container for the experiment run"""
        return PostprocessedExperimentData(
            experiment=self,
            postprocessors=[('set_child_data_structure', self.set_child_data_structure)]
        )

    @classmethod
    def set_child_data_structure(
        cls,
        experiment_data: ExperimentData
    ) -> None:
        """Recursively create data parent-child hierarchy for CompositeExperiment results."""
        #composite_analysis._add_child_data(experiment_data)
        cls._add_child_data(experiment_data)
        for child_data in experiment_data.child_data():
            if 'component_types' in child_data.metadata:
                cls.set_child_data_structure(child_data)

    @classmethod
    def fill_child_data(
        cls,
        experiment_data: ExperimentData
    ):
        """Recursively fill the child data container using a method in CompositeAnalysis."""
        num_children = len(experiment_data.metadata['component_types'])
        CompositeAnalysis([None] * num_children)._component_experiment_data(experiment_data)
        for child_data in experiment_data.child_data():
            if 'component_types' in child_data.metadata:
                cls.fill_child_data(child_data)

    @classmethod
    def _add_child_data(
        cls,
        experiment_data: ExperimentData
    ):
        """Reimplementing add_child_data to avoid needlessly instantiating ExperimentData without
        the service keyword.
        """
        component_index = experiment_data.metadata.get("component_child_index", [])
        if component_index:
            # Child components are already initialized
            return

        # Initialize the component experiment data containers and add them
        # as child data to the current experiment data
        child_components = cls._initialize_component_experiment_data(experiment_data)
        start_index = len(experiment_data.child_data())
        for i, subdata in enumerate(child_components):
            experiment_data.add_child_data(subdata)
            component_index.append(start_index + i)

        # Store the indices of the added child data in metadata
        experiment_data.metadata["component_child_index"] = component_index

    @classmethod
    def _initialize_component_experiment_data(
        cls,
        experiment_data: ExperimentData
    ) -> list[ExperimentData]:
        """Set the child data structure."""
        experiment_types = experiment_data.metadata["component_types"]
        component_metadata = experiment_data.metadata["component_metadata"]

        # Create component experiments and set the backend and metadata for the components
        component_expdata = []
        for exp_type, metadata in zip(experiment_types, component_metadata):
            # We needed to reimplement add_child_data and this function just for this line
            subdata = ExperimentData(backend=experiment_data.backend,
                                     service=experiment_data.service)
            subdata.experiment_type = exp_type
            subdata.metadata.update(metadata)

            # Copy tags, share_level and auto_save from the parent
            # experiment data if results are not being flattened.
            subdata.tags = experiment_data.tags
            subdata.share_level = experiment_data.share_level
            subdata.auto_save = experiment_data.auto_save

            component_expdata.append(subdata)

        return component_expdata
