from pygrank.algorithms.filters.abstract import ClosedFormGraphFilter


class HeatKernel(ClosedFormGraphFilter):
    """Heat kernel filter."""

    def __init__(self, t: float = 3, *args, **kwargs):
        """Initializes filter parameters.

        Args:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.

        Example:
            >>> from pygrank.algorithms import HeatKernel
            >>> algorithm = HeatKernel(t=3, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.t = t
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        # backend.exp(-self.t)
        return (
            1.0
            if previous_coefficient is None
            else (previous_coefficient * self.t / (self.convergence.iteration + 1))
        )

    def references(self):
        refs = super().references()
        refs[0] = "HeatKernel \\cite{chung2007heat}"
        refs.insert(1, f"emphasis on {self.t}-hop distances")
        return refs
