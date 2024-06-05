from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class Cos(Supervised):
    """Computes the cosine similarity between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        divide = backend.dot(known_scores, known_scores) * backend.dot(scores, scores)
        return backend.safe_div(backend.dot(known_scores, scores), divide**0.5)
