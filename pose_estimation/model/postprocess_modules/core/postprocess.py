from abc import abstractmethod, ABC


class InterfacePostProcessModule(ABC):

    @abstractmethod
    def set_paf_heatmap(self, paf, heatmap):
        pass

    @abstractmethod
    def set_session(self, session):
        pass

    @abstractmethod
    def __call__(self, feed_dict):
        pass

