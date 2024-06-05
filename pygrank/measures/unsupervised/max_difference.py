from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class MaxDifference(Supervised):
    """Computes the maximum absolute error between scores and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return backend.max(backend.abs(known_scores - scores))
