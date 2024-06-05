from pygrank.measures.unsupervised.unsupervised import Unsupervised
import numpy as np
from pygrank.core.signals import to_signal
from pygrank.core import GraphSignalGraph, GraphSignalData, BackendPrimitive


class Modularity(Unsupervised):
    """
    Extension of modularity that accounts for node scores.
    """

    def __init__(
        self,
        graph: GraphSignalGraph = None,
        max_rank: float = 1,
        max_positive_samples: int = 2000,
        seed: int = 0,
        progress=lambda x: x,
    ):
        """Initializes the Modularity measure with a sampling strategy that speeds up normal computations.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically
             extracted from graph signals passed for evaluation.
            max_rank: Optional. Default is 1.
            max_positive_samples: Optional. The number of nodes with which to compute modularity. These are
             sampled uniformly from all graph nodes. If this is greater than the number of graph nodes,
             all nodes are used and the measure is deterministic. However,
             calculation time is O(max_positive_samples<sup>2</sup>) and thus a trade-off needs to be determined of time
             vs approximation quality. Effectively, the value should be high enough for max_positive_samples<sup>2</sup>
             to be comparable to the number of graph edges. Default is 2000.
            seed: Optional. Makes the evaluation seeded, for example to use in tuning. Default is 0.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> modularity = pg.Modularity(max_positive_samples=int(graph.number_of_edges()**0.5)).evaluate(scores)
        """
        self.graph = graph
        self.max_positive_samples = max_positive_samples
        self.max_rank = max_rank
        self.seed = seed
        self.progress = progress

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        scores = to_signal(self.graph, scores)
        graph = scores.graph
        positive_candidates = list(graph)
        if len(positive_candidates) > self.max_positive_samples:
            np.random.seed(self.seed)
            positive_candidates = np.random.choice(
                positive_candidates, self.max_positive_samples
            )
        m = graph.number_of_edges()
        if m == 0:
            return 0
        Q = 0
        for v in self.progress(positive_candidates):
            for u in positive_candidates:
                Avu = 1 if graph.has_edge(v, u) else 0
                Avu -= graph.degree[v] * graph.degree[u] / 2 / m
                Q += Avu * (scores[v] / self.max_rank) * (scores[u] / self.max_rank)
        return Q / 2 / m
