from pygrank.algorithms.filters.abstract import ClosedFormGraphFilter


class PageRankClosed(ClosedFormGraphFilter):
    """PageRank closed filter."""

    def __init__(self, alpha: float = 0.85, *args, **kwargs):
        """Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default alpha value is 0.85
                for historyical reasons. However, in large graphs it is often preferred to set this
                argument to 0.9.
        Example:
            >>> import pygrank as pg
            >>> algorithm = pg.PageRankClosed(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        return (
            1.0 if previous_coefficient is None else (previous_coefficient * self.alpha)
        )

    def references(self):
        refs = super().references()
        refs[0] = "polynomial personalized PageRank \\cite{page1999pagerank}"
        refs.insert(1, f"diffusion rate {self.alpha:.3f}")
        return refs
