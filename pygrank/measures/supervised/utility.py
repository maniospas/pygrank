from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import GraphSignalData


class Utility:
    def __init__(
        self, measure: Supervised, transform=lambda x: x, transform_exclude=lambda x: x
    ):
        self.measure = measure
        self.transform = transform
        self.transform_exclude = transform_exclude
        self.__name__ = measure.__name__ + " Utility"

    def __call__(
        self, known_scores: GraphSignalData, exclude: GraphSignalData, *args, **kwargs
    ):
        return self.measure(
            self.transform(exclude), self.transform_exclude(exclude), *args, **kwargs
        )
