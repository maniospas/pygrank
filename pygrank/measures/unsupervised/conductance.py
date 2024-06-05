from pygrank.measures.unsupervised.unsupervised import Unsupervised
from pygrank.core import backend, GraphSignalGraph, GraphSignalData, BackendPrimitive


class Conductance(Unsupervised):
    """Graph conductance (information flow) of scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of
    the fuzzy rank subgraph. To avoid potential optimization towards filling the whole graph, the measure is
    evaluated to infinity if either denominator *or* the nominator is zero (this means that whole connected components
    should not be extracted).
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance.
    """

    def __init__(
        self,
        graph: GraphSignalGraph = None,
        max_rank: float = 1,
        autofix=False,
        cut_ratio_only=False,
        **kwargs
    ):
        """Initializes the Conductance measure.

        Args:
            max_rank: Optional. The maximum value scores can assume. To maintain a probabilistic formulation of
             conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1.
             Pass algorithms through a normalization to ensure that this limit is not violated.
            autofix: Optional. If True, automatically normalizes scores by multiplying with max_rank / their maximum.
             If False (default) and the maximum score is greater than max_rank, an exception is thrown.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> algorithm = pg.Normalize(algorithm)
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Conductance().evaluate(scores)

        Example (same conductance):
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Conductance(autofix=True).evaluate(scores)
        """
        super().__init__(graph, **kwargs)
        self.max_rank = max_rank
        self.autofix = autofix
        self.cut_ratio_only = cut_ratio_only

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        graph = self.get_graph(scores)
        if len(graph) == 0:
            return float("inf")
        adjacency, scores = self.to_numpy(scores)
        if backend.max(scores) > self.max_rank:
            if self.autofix:
                scores = scores * (self.max_rank / backend.max(scores))
            else:
                raise Exception(
                    "Normalize scores to be <= "
                    + str(self.max_rank)
                    + " for non-negative conductance"
                )
        neighbors = backend.conv(scores, adjacency)
        internal_edges = backend.dot(neighbors, scores)
        if not self.cut_ratio_only:
            internal_edges = min(
                internal_edges,
                backend.dot(
                    backend.conv(self.max_rank - scores, adjacency),
                    self.max_rank - scores,
                ),
            )
        external_edges = backend.dot(neighbors, self.max_rank - scores)
        if not graph.is_directed():
            external_edges += backend.dot(
                scores, backend.conv(self.max_rank - scores, adjacency)
            )
            internal_edges *= 2
        if external_edges == 0:
            return float("inf")
        return backend.safe_div(external_edges, internal_edges, default=float("inf"))
