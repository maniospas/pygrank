from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class TNR(Supervised):
    """Computes the false negative rate."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        known_scores = backend.safe_div(known_scores, backend.max(known_scores))
        scores = backend.safe_div(scores, backend.max(scores))
        return backend.safe_div(
            backend.sum((1 - known_scores) * (1 - scores)),
            backend.sum(1 - known_scores),
        )
