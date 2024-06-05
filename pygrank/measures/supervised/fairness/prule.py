from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class pRule(Supervised):
    """Computes an assessment of stochastic ranking fairness by obtaining the fractional comparison of average scores
    between sensitive-attributed nodes and the rest the rest.
    Values near 1 indicate full fairness (statistical parity), whereas lower values indicate disparate impact.
    Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive.
    Usually, pRule > 80% is considered fair.
    """

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive, scores = self.to_numpy(scores)
        p1 = backend.dot(scores, sensitive)
        p2 = backend.dot(scores, 1 - sensitive)
        if p1 == 0 or p2 == 0:
            return 0
        s = backend.sum(sensitive)
        p1 = backend.safe_div(p1, s)
        p2 = backend.safe_div(p2, backend.length(sensitive) - s)
        p1 = backend.abs(p1)
        p2 = backend.abs(p2)
        if p1 <= p2:  # this implementation is derivable
            return p1 / p2
        return p2 / p1
