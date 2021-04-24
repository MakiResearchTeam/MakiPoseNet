from ..core import SkeletonCorrector


class Identity(SkeletonCorrector):
    """
    Does nothing and just returns the given skeletons.
    """
    def __call__(self, skeletons: list) -> list:
        """
        Return input `skeletons` as it is
        Without even touch them

        """
        return skeletons

