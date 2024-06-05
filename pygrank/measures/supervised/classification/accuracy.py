from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class Accuracy(Supervised):
    """Computes the accuracy as 1- mean absolute differences between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return 1 - backend.sum(backend.abs(known_scores - scores)) / backend.length(
            scores
        )
