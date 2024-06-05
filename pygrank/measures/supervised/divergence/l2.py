from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class L2(Supervised):
    """Computes the L2 norm on the difference between scores and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return backend.sum((known_scores - scores) * (known_scores - scores))
