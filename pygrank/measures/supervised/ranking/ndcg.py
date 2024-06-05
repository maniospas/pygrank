from pygrank.measures.supervised.supervised import Supervised
import numpy as np
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class NDCG(Supervised):
    """Provides evaluation of NDCG@k score between given and known scores."""

    def __init__(
        self,
        known_scores: GraphSignalData,
        exclude: GraphSignalData = None,
        k: int = None,
    ):
        """Initializes the supervised measure with desired graph signal outcomes and the number of top scores.

        Args:
            k: Optional. Calculates NDCG@k. If None (default), len(known_scores) is used.
        """
        super().__init__(known_scores, exclude=exclude)
        if k is not None and k > len(known_scores):
            raise Exception(
                "NDCG@k cannot be computed for k greater than the number of known scores"
            )
        self.k = len(known_scores) if k is None else k

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        DCG = 0
        IDCG = 0
        for i, v in enumerate(
            list(
                sorted(
                    list(range(backend.length(scores))),
                    key=scores.__getitem__,
                    reverse=True,
                )
            )[: self.k]
        ):
            DCG += known_scores[v] / np.log2(i + 2)
        for i, v in enumerate(
            list(
                sorted(
                    list(range(backend.length(known_scores))),
                    key=known_scores.__getitem__,
                    reverse=True,
                )
            )[: self.k]
        ):
            IDCG += known_scores[v] / np.log2(i + 2)
        return DCG / IDCG
