from ..core import SkeletonCorrector


class CorrectorCompose(SkeletonCorrector):
    def __init__(self, correctors: list):
        """
        Create a composition of several correctors that will be applied
        to the skeletons one after another.

        Parameters
        ----------
        correctors : list
            A list of correctors to apply.
        """
        self._correctors = correctors

    def __call__(self, skeletons):
        for corrector in self._correctors:
            skeletons = corrector(skeletons)
        return skeletons
