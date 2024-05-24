"""Functions to construct and fill child data independently from analyses."""
from qiskit_experiments.framework import CompositeAnalysis, ExperimentData


def set_child_data_structure(experiment_data: ExperimentData):
    """Recursively create data parent-child hierarchy for CompositeExperiment results."""
    # composite_analysis._add_child_data(experiment_data)
    _add_child_data(experiment_data)
    for child_data in experiment_data.child_data():
        if 'component_types' in child_data.metadata:
            set_child_data_structure(child_data)


def fill_child_data(experiment_data: ExperimentData):
    """Recursively fill the child data container using a method in CompositeAnalysis."""
    num_children = len(experiment_data.metadata['component_types'])
    CompositeAnalysis([None] * num_children)._component_experiment_data(experiment_data)
    for child_data in experiment_data.child_data():
        if 'component_types' in child_data.metadata:
            fill_child_data(child_data)


def _add_child_data(experiment_data: ExperimentData):
    """Reimplementing add_child_data to avoid needlessly instantiating ExperimentData without the
    service keyword.
    """
    component_index = experiment_data.metadata.get("component_child_index", [])
    if component_index:
        # Child components are already initialized
        return
    # Initialize the component experiment data containers and add them
    # as child data to the current experiment data
    child_components = _initialize_component_experiment_data(experiment_data)
    start_index = len(experiment_data.child_data())
    for i, subdata in enumerate(child_components):
        experiment_data.add_child_data(subdata)
        component_index.append(start_index + i)
    # Store the indices of the added child data in metadata
    experiment_data.metadata["component_child_index"] = component_index


def _initialize_component_experiment_data(experiment_data: ExperimentData) -> list[ExperimentData]:
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
