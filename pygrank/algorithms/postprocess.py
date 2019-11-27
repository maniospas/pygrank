import warnings


class Tautology:
    """ Returns ranks as-are.

    This class can be used as a baseline against which to compare other rank post-processing algorithms
    (e.g. those of this package).
    """

    def __init__(self):
        pass

    def rank(self, _, personalization):
        return personalization


class Normalize:
    """ Normalizes ranks by dividing with their maximal value."""

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> algorithm = Threshold(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(G, seed_values)

        Example (same outcome, quicker one-time use):
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> G, seed_values, algorithm = ...
            >>> ranks = Normalize(0.5).transform(algorithm.rank(G, seed_values))
        """
        self.ranker = Tautology() if ranker is None else ranker

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        max_rank = max(ranks.values())
        return {node: rank / max_rank for node, rank in ranks.items()}

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        return self.transform(ranks)


class Ordinals:
    """ Converts ranking outcome to ordinal numbers.

    The highest rank is set to 1, the second highest to 2, etc.
    """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
        """
        self.ranker = Tautology() if ranker is None else ranker

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        return {v: ord+1 for ord, v in enumerate(sorted(ranks, key=ranks.get, reverse=False))}

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        return self.transform(ranks)


class Threshold:
    """ Converts ranking outcome to binary values based on a threshold value."""

    def __init__(self, threshold="gap", ranker=None):
        """ Initializes the Threshold postprocessing scheme.

        Attributes:
            threshold: Optional. The minimum numeric value required to output rank 1 instead of 0. If "gap" (default)
                then its value is automatically determined based on the maximal percentage increase between consecutive
                ranks.
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> algorithm = Threshold(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(G, seed_values)

        Example (same outcome):
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> ranks = Threshold(0.5).transform(algorithm.rank(G, seed_values))
        """
        self.ranker = Tautology() if ranker is None else ranker
        self.threshold = threshold

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        threshold = self.threshold
        if threshold=="gap":
            warnings.warn("gap-determined threshold is still under development (its implementation may be incorrect)", stacklevel=2)
            ranks = {v: ranks[v] / self.G.degree(v) for v in ranks}
            max_diff = 0
            threshold = 0
            prev_rank = 0
            for v in sorted(ranks, key=ranks.get, reverse=True):
                if prev_rank > 0:
                    diff = (prev_rank - ranks[v]) / prev_rank
                    if diff > max_diff:
                        max_diff = diff
                        threshold = ranks[v]
                prev_rank = ranks[v]
        return self.base_metric.evaluate({v: 1 for v in ranks.keys() if ranks[v] >= threshold})

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        return self.transform(ranks)