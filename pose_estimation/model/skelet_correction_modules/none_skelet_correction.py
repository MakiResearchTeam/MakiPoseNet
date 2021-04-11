from ..core import SkeletonCorrector


class SkeletCorrectionNoneModule(SkeletonCorrector):
    """
    This class itself, does nothing
    """
    def __call__(self, skeletons: list) -> list:
        """
        Return input `skeletons` as it is
        Without even touch them

        """
        return skeletons

