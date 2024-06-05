from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter
from pygrank.core import to_signal


class AbsorbingWalks(RecursiveGraphFilter):
    """Implementation of partial absorbing random walks for Lambda = (1-alpha)/alpha diag(absorption vector).
    To determine parameters based on symmetricity principles, use *SymmetricAbsorbingRandomWalks*.
    """

    def __init__(self, alpha: float = 1 - 1.0e-6, *args, **kwargs):
        """Initializes filter parameters. The filter can model PageRank for appropriate parameter values,
        but is in principle a generalization that allows custom absorption rates per node (when not given, these are I).

        Args:
            alpha: Optional. (1-alpha)/alpha is the absorption rate of the random walk multiplied with individual node
                absorption rates. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = alpha Diag(degrees) ) when the same parameter
                value alpha is chosen. Default is 1-1.E-6 per the respective publication.

        Example:
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})

        Example (same outcome, explicit absorption rate definition):
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes}, absorption={v: 1 for v in graph})
        """

        super().__init__(*args, **kwargs)
        self.alpha = alpha  # typecast to make sure that a graph is not accidentally the first argument

    def _start(self, M, personalization, ranks, absorption=None, **kwargs):
        self.absorption = to_signal(personalization.graph, absorption) * (
            (1 - self.alpha) / self.alpha
        )
        self.degrees = backend.degrees(M)

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        ret = (
            backend.conv(ranks, M) * self.degrees + personalization * self.absorption
        ) / (self.absorption + self.degrees)
        return ret

    def references(self):
        refs = super().references()
        refs[0] = "partially absorbing random walks \\cite{wu2012learning}"
        return refs
