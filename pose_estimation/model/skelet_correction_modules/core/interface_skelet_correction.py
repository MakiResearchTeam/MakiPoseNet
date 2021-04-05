from abc import abstractmethod, ABC


class InterfaceSkeletCorrectionModule(ABC):

    @abstractmethod
    def __call__(self, skeletons: list) -> list:
        pass

