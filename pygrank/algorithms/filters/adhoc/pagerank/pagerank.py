from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter


class PageRank(RecursiveGraphFilter):
    """A Personalized PageRank power method algorithm."""

    def __init__(self, alpha: float = 0.85, *args, **kwargs):
        """Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default alpha value is 0.85
                for historyical reasons. However, in large graphs it is often preferred to set this
                argument to 0.9.
        Example:
            >>> import pygrank as pg
            >>> algorithm = pg.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def _start(self, M, personalization, ranks, *args, **kwargs):
        # self.dangling_weights = backend.degrees(M)
        # self.is_dangling = self.dangling_weights/backend.sum(self.dangling_weights)
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        # TODO: return self.alpha * (ranks * M + backend.sum(ranks[self.is_dangling]) * personalization) + (1 - self.alpha) * personalization
        return backend.conv(ranks, M) * self.alpha + personalization * (1 - self.alpha)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        # del self.is_dangling
        super()._end(M, personalization, ranks, *args, **kwargs)

    def references(self):
        refs = super().references()
        refs[0] = "personalized PageRank \\cite{page1999pagerank}"
        refs.insert(1, f"diffusion rate {self.alpha:.3f}")
        return refs
