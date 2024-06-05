from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class BinaryCrossEntropy(Supervised):
    """Computes a cross-entropy loss of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        # thresh = backend.min(scores[known_scores!=0])
        # scores = 1/(1+np.exp(-scores/thresh+1))
        eps = backend.epsilon()
        ret = -backend.dot(known_scores, backend.log(scores + eps)) - backend.dot(
            1 - known_scores, backend.log(1 - scores + eps)
        )
        return ret
