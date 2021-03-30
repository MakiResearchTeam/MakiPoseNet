# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from abc import abstractmethod, ABC


class InterfacePostProcessModule(ABC):

    DEFAULT_SCALE = 8

    def __init__(self):
        self._resize_to = None
        self._session = None
        self._heatmap_tensor = None
        self._paf_tensor = None
        self._is_graph_build = False
    
    def get_paf_tensor(self):
        """
        Return paf tensor
        NOTICE! If its not set - will be dropped error

        """
        if self._paf_tensor is None:
            raise ValueError("paf_tensor parameter was not set.")
        
        return self._paf_tensor
    
    def get_heatmap_tensor(self):
        """
        Return heatmap tensor
        NOTICE! If its not set - will be dropped error

        """
        if self._heatmap_tensor is None:
            raise ValueError("heatmap parameter was not set.")

        return self._heatmap_tensor
    
    def set_paf_heatmap(self, paf, heatmap):
        """
        Set two tf.Tensor of paf and heatmap into this class
        If any value is None, set will be skipped

        """
        assert paf is not None, "Input paf is None, but must be of type tf.Tensor/tf.Variable"
        assert heatmap is not None, "Input heatmap is None, but must be of type tf.Tensor/tf.Variable"

        self._paf_tensor = paf
        self._heatmap_tensor = heatmap

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

    def set_resize_to(self, resize_to: tuple):
        """
        Set new value for resize_to parameter

        Parameters
        ----------
        resize_to : tuple
            (H, W) tuple of Height and Width

        """
        self._resize_to = resize_to

    def set_session(self, session):
        """
        Set session to postprocess class

        Parameters
        ----------
        session : tf.Session
            Instance of class tf.Session

        """
        self._session = session

    def __call__(self, feed_dict):
        """
        Gives paf, indices and peaks according to input `feed_dict`

        Parameters
        ----------
        feed_dict : dict
            Example: { placholder: np.ndarray }, which further calls with session

        Returns
        -------
        paf : np.ndarray
        indices : np.ndarray
        peaks : np.ndarray

        """
        self.compile()
        return self._execute_postprocess(feed_dict=feed_dict)

    @abstractmethod
    def _build_postporcess_graph(self):
        """
        Method build necessary tf graph for further execution in _execute_postprocess method

        """
        pass

    @abstractmethod
    def _execute_postprocess(self, feed_dict):
        """
        Execute postprocess operations

        """
        pass

    def get_data_for_debug(self, feed_dict):
        raise NotImplemented("This method should not be called from this class.")

    def compile(self):
        """
        Build post-graph if its not will be build

        """
        if not self._is_graph_build:
            self._build_postporcess_graph()
            self._is_graph_build = True
