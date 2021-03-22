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

import cv2
import numpy as np


class CPUOptimizedPostProcessNPPart:

    def __init__(self, resize_to, upsample_heatmap=False):
        self.__resize_to = resize_to
        self.__upsample_heatmap = upsample_heatmap
        self._saved_mesh_grid = None

    def set_resize_to(self, new_resize_to):
        self.__resize_to = new_resize_to

    def process(self, heatmap, paf):
        upsample_paf = self._process_paf(paf)
        indices, peaks = self._process_heatmap(heatmap)
        return upsample_paf, indices, peaks

    def _process_heatmap(self, heatmap):
        heatmap = heatmap[0]
        if self.__upsample_heatmap:
            heatmap = cv2.resize(
                heatmap,
                (self.__resize_to[1], self.__resize_to[0]),
                interpolation=cv2.INTER_LINEAR
            )
        indices, peaks = self._apply_nms_and_get_indices(heatmap)

        return indices, peaks

    def _process_paf(self, paf):
        h_f, w_f = paf[0].shape[:2]
        paf_pr = cv2.resize(
            paf[0].reshape(h_f, w_f, -1),
            (self.__resize_to[1], self.__resize_to[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return paf_pr

    def _apply_nms_and_get_indices(self, heatmap_pr):
        heatmap_pr[heatmap_pr < 0.1] = 0
        heatmap_with_borders = np.pad(heatmap_pr, [(2, 2), (2, 2), (0, 0)], mode='constant')
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 1:heatmap_with_borders.shape[1] - 1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 0:heatmap_with_borders.shape[1] - 2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1] - 1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] - 2, 1:heatmap_with_borders.shape[1] - 1]

        heatmap_peaks = (heatmap_center > heatmap_left) & \
                        (heatmap_center > heatmap_right) & \
                        (heatmap_center > heatmap_up) & \
                        (heatmap_center > heatmap_down)

        indices, peaks = self._get_peak_indices(heatmap_peaks)

        return indices, peaks

    def _get_peak_indices(self, array):
        """
        Returns array indices of the values larger than threshold.
        Parameters
        ----------
        array : ndarray of any shape
            Tensor which values' indices to gather.
        Returns
        -------
        ndarray of shape [n_peaks, dim(array)]
            Array of indices of the values larger than threshold.
        ndarray of shape [n_peaks]
            Array of the values at corresponding indices.
        """
        flat_peaks = np.reshape(array, -1)
        if self._saved_mesh_grid is None or len(flat_peaks) != self._saved_mesh_grid.shape[0]:
            self._saved_mesh_grid = np.arange(len(flat_peaks))

        peaks_coords = self._saved_mesh_grid[flat_peaks]
        peaks = np.ones(len(peaks_coords), dtype=np.float32)
        indices = np.unravel_index(peaks_coords, shape=array.shape)
        indices = np.stack(indices, axis=-1).astype(np.int32, copy=False)
        return indices, peaks

