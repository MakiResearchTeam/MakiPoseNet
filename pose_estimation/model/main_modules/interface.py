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

from abc import ABC, abstractmethod
from makiflow.base.maki_entities.maki_core import MakiCore


class PoseEstimatorInterface(MakiCore, ABC):

    @abstractmethod
    def training_on(self):
        pass

    @abstractmethod
    def predict(self, x: list):
        pass

    @abstractmethod
    def get_paf_tensors(self) -> list:
        pass

    @abstractmethod
    def get_heatmap_tensors(self) -> list:
        pass

    @abstractmethod
    def get_training_vars(self):
        pass

    @abstractmethod
    def build_final_loss(self, loss):
        pass

    @abstractmethod
    def get_session(self):
        # TODO: Move this method into MakiModel
        pass

    @abstractmethod
    def get_paf_makitensor(self):
        pass

    @abstractmethod
    def get_heatmap_makitensor(self):
        pass

    @abstractmethod
    def get_batch_size(self):
        pass
