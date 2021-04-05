from pose_estimation.model.skelet_correction_modules.core import InterfaceSkeletCorrectionModule


class SkeletCorrectionNoneModule(InterfaceSkeletCorrectionModule):
    """
    This class itself, does nothing

    """

    def __call__(self, skeletons: list) -> list:
        """
        Return input `skeletons` as it is
        Without even touch them

        """
        return skeletons

