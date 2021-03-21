from abc import abstractmethod, ABC


class InterfacePostProcessModule(ABC):

    def __init__(self):
        self._resize_to = None
        self._session = None

    @abstractmethod
    def set_paf_heatmap(self, paf, heatmap):
        pass

    def get_resize_to(self):
        """
        Returns
        -------
        resize_to : tuple
            Tuple of (H, W), Height and Width

        """
        if self._resize_to is None:
            raise ValueError("resize_to parameter was not set.")

        return self._resize_to

    def set_session(self, session):
        """
        Set session to postprocess class

        Parameters
        ----------
        session : tf.Session
            Instance of class tf.Session

        """
        self._session = session

    def set_resize_to(self, resize_to: tuple):
        """
        Set new value for resize_to parameter

        Parameters
        ----------
        resize_to : tuple
            (H, W) tuple of Height and Width

        """
        self._resize_to = resize_to

    @abstractmethod
    def __call__(self, feed_dict):
        pass


