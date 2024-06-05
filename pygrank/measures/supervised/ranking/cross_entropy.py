from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class CrossEntropy(Supervised):
    """Computes the KL-divergence of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores, normalization=False)
        ret = -backend.sum(scores * backend.log(known_scores))
        return ret
