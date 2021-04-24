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

import tensorflow as tf


class MakiLayerWrapper:
    """
    Simple MakiLayer wrapper
    in order to use them after we build label correction graph and hide tf.Tensor inside this class
    And mimic MakiLayer
    """

    def __init__(self, data_tensor: tf.Tensor):
        """

        Parameters
        ----------
        data_tensor : tf.Tensor
            Tensor that will be stored in this class

        """
        self._data_tensor = data_tensor

    def get_data_tensor(self) -> tf.Tensor:
        """
        Returns
        -------
        tf.Tensor
            Stored data tensor

        """
        return self._data_tensor
