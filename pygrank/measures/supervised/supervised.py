from pygrank.measures.measure import Measure
from pygrank.core import (
    backend,
    GraphSignal,
    to_signal,
    GraphSignalData,
    BackendPrimitive,
)
import numbers
from typing import Tuple, Union


class Supervised(Measure):
    """Provides a base class with the ability to simultaneously convert scores and known scores to numpy arrays.
    This class is used as a base for other supervised evaluation measures."""

    def __init__(self, known_scores: GraphSignalData, exclude: GraphSignalData = None):
        """
        Initializes the supervised measure with desired graph signal outcomes.
        Args:
            known_scores: The desired graph signal outcomes.
            exclude: Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed
                to determine which nodes to ommit from the evaluation, for example because they were used for training.
                If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude`
                property at any time to alter this original value. Prefer using this behavior to avoid overfitting
                measure assessments.
        """
        self.known_scores = known_scores
        self.exclude = exclude

    def to_numpy(
        self, scores: GraphSignalData, normalization: bool = False
    ) -> Union[
        Tuple[GraphSignal, GraphSignal], Tuple[BackendPrimitive, BackendPrimitive]
    ]:
        if isinstance(scores, numbers.Number) and isinstance(
            self.known_scores, numbers.Number
        ):
            return backend.to_array([self.known_scores]), backend.to_array([scores])
        elif isinstance(scores, GraphSignal):
            return to_signal(scores, self.known_scores).filter(
                exclude=self.exclude
            ), scores.normalized(normalization).filter(exclude=self.exclude)
        elif isinstance(self.known_scores, GraphSignal):
            return self.known_scores.filter(exclude=self.exclude), to_signal(
                self.known_scores, scores
            ).normalized(normalization).filter(exclude=self.exclude)
        else:
            if self.exclude is not None:
                raise Exception(
                    "Needs to parse graph signal scores or known_scores to be able to exclude specific nodes"
                )
            scores = (
                backend.self_normalize(backend.to_array(scores, copy_array=True))
                if normalization
                else backend.to_array(scores)
            )
            return backend.to_array(self.known_scores), scores

    def best_direction(self) -> int:
        ret = getattr(self.__class__, "__best_direction", None)
        if ret is None:
            ret = (
                1
                if self.__class__([1, 0])([1, 0]) > self.__class__([1, 0])([0, 1])
                else -1
            )
            setattr(self.__class__, "__best_direction", ret)
        return ret
