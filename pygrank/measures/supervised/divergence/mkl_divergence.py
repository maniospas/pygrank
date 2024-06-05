from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class MKLDivergence(Supervised):
    """Computes the mean KL-divergence of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores, normalization=True)
        eps = backend.epsilon()
        known_scores = known_scores + eps
        known_scores = known_scores / backend.sum(known_scores)
        scores = scores + eps
        scores = scores / backend.sum(scores)
        ratio = scores / known_scores
        ret = -backend.sum(scores * backend.log(ratio))
        return ret / backend.length(scores)
