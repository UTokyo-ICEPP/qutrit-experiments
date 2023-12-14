"""ExperimentData with postprocessors."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Union

from qiskit_experiments.framework import BaseExperiment, ExperimentData, AnalysisStatus
from qiskit_experiments.database_service.exceptions import ExperimentDataError

if TYPE_CHECKING:
    from qiskit.providers import Job, Backend
    from qiskit_ibm_experiment import IBMExperimentService, ExperimentData as ExperimentDataclass

PostProcessor = tuple[str, Callable[[ExperimentData], None]]


class PostprocessedExperimentData(ExperimentData):
    """An ExperimentData subclass with data post-processing callbacks."""

    def __init__(
        self,
        experiment: Optional[BaseExperiment] = None,
        backend: Optional['Backend'] = None,
        service: Optional['IBMExperimentService'] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[list[str]] = None,
        child_data: Optional[list[ExperimentData]] = None,
        verbose: Optional[bool] = True,
        db_data: Optional['ExperimentDataclass'] = None,
        postprocessors: Optional[list[PostProcessor]] = None,
        **kwargs,
    ):
        super().__init__(
            experiment=experiment,
            backend=backend,
            service=service,
            parent_id=parent_id,
            job_ids=job_ids,
            child_data=child_data,
            verbose=verbose,
            db_data=db_data,
            **kwargs
        )

        if postprocessors:
            self._postprocessors = list(postprocessors)
            self._postprocessor_ids = {name: None for name, _ in postprocessors}
        else:
            self._postprocessors = []
            self._postprocessor_ids = {}

    def add_postprocessor(
        self,
        name: str,
        callback: Callable[[ExperimentData], None],
        position: Optional[int] = None
    ):
        if position is None:
            self._postprocessors.append((name, callback))
        else:
            self._postprocessors.insert(position, (name, callback))

        self._postprocessor_ids[name] = None

    def apply_postprocessors(self):
        for name, callback in self._postprocessors:
            with self._analysis_callbacks.lock:
                cids = set(self._analysis_callbacks.keys())
                self.add_analysis_callback(callback)
                new_cid = list(set(self._analysis_callbacks.keys()) - cids)[0]
                self._postprocessor_ids[name] = new_cid

    def postprocessor_status(
        self,
        name: Optional[str] = None
    ) -> Union[AnalysisStatus, dict[str, AnalysisStatus]]:
        if name is None:
            return {name: self._analysis_callbacks[cid].status
                    for name, cid in self._postprocessor_ids.items() if cid is not None}

        try:
            cid = self._postprocessor_ids[name]
            if not cid:
                raise KeyError(name)
        except KeyError as exc:
            raise ValueError(f'Status is undefined for postprocessor {name}') from exc

        return self._analysis_callbacks[cid].status

    def block_for_results(
        self,
        timeout: Optional[float] = None
    ) -> "PostprocessedExperimentData":
        super().block_for_results(timeout=timeout)

        ## Check the postprocessor statuses
        for name, status in self.postprocessor_status().items():
            if status != AnalysisStatus.DONE:
                raise ExperimentDataError(f'Postprocessor {name} status = {status.value}')

        return self
