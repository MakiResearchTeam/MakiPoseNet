from abc import abstractmethod, ABC


class InterfacePostProcessModule(ABC):

    @abstractmethod
    def set_paf_heatmap(self, paf, heatmap):
        pass

    @abstractmethod
    def set_session(self, session):
        pass

    @abstractmethod
    def __call__(self, input_batch, feed_dict, using):
        pass

    @abstractmethod
    def _build_postporcess_graph(self):
        pass

    @abstractmethod
    def _process_not_tf_part(self, **kwargs):
        pass

