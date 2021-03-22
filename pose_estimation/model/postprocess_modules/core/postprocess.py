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
        self._is_using_estimate_alg = True
    
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
        if paf is not None:
            self._paf_tensor = paf

        if heatmap is not None:
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

    def set_is_using_estimate_alg(self, is_use: bool):
        self._is_using_estimate_alg = is_use

    def get_is_using_estimate_alg(self):
        return self._is_using_estimate_alg

    def set_session(self, session):
        """
        Set session to postprocess class

        Parameters
        ----------
        session : tf.Session
            Instance of class tf.Session

        """
        self._session = session

    @abstractmethod
    def __call__(self, feed_dict):
        pass


