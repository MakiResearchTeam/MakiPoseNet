from abc import abstractmethod, ABC


class SkeletonCorrector(ABC):

    @abstractmethod
    def __call__(self, skeletons) -> list:
        pass

