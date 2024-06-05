from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter


class DijkstraRank(RecursiveGraphFilter):
    """A ranking algorithm that assigns node ranks loosely increasing with the minimum distance from a seed."""

    def __init__(self, degradation=0.1, *args, **kwargs):
        self.degradation = degradation
        super().__init__(*args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        prev_ranks = ranks
        ranks = backend.conv(ranks, M) * self.degradation
        choice = backend.cast(ranks.np > prev_ranks.np)
        ranks.np = choice * ranks.np + (1 - choice) * prev_ranks.np
        return ranks
