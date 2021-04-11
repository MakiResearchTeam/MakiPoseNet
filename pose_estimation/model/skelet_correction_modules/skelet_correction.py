from ..core import SkeletonCorrector


class SkeletCorrectionOneEuroModule(SkeletonCorrector):
    """
    This correction module is based in 1 euro algorithm
    For mode details refer to: https://hal.inria.fr/hal-00670496/document
    """

    def __init__(self, alpha=0.9):
        self._alpha = alpha

    def __call__(self, skeletons: list) -> list:
        pass
