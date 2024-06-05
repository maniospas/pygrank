from pygrank.measures.unsupervised.unsupervised import Unsupervised
from pygrank.core import backend, GraphSignalGraph, GraphSignalData, BackendPrimitive


class Density(Unsupervised):
    """Extension of graph density that accounts for node scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of
    the fuzzy rank subgraph.
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Density.
    """

    def __init__(self, graph: GraphSignalGraph = None, **kwargs):
        """Initializes the Density measure.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically
             extracted from graph signals passed for evaluation.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> density = pg.Density().evaluate(scores)
        """
        super().__init__(graph, **kwargs)

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        if len(self.get_graph(scores)) == 0:
            return 0
        adjacency, scores = self.to_numpy(scores)
        neighbors = backend.conv(scores, adjacency)
        internal_edges = backend.dot(neighbors, scores)
        expected_edges = backend.sum(scores) ** 2 - backend.sum(
            scores**2
        )  # without self-loops
        return backend.safe_div(internal_edges, expected_edges)
