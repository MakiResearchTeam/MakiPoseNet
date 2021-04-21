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
from .constants import CONNECT_INDEXES
from ...model.utils.human import Human

EPSILONE = 1e-2


def visualize_paf(
        img,
        pafs,
        colors=((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0))):
    img = img.copy()
    color_iter = 0
    for i in range(pafs.shape[2]):
        paf_x = pafs[:, :, i, 0]
        paf_y = pafs[:, :, i, 1]
        len_paf = np.sqrt(paf_x ** 2 + paf_y ** 2)
        for x in range(0, img.shape[0], 8):
            for y in range(0, img.shape[1], 8):
                if len_paf[x,y] > 0.1:
                    img = cv2.arrowedLine(
                        img,
                        (y, x),
                        (int(y + 9 * paf_x[x, y]), int(x + 9 * paf_y[x, y])),
                        colors[color_iter],
                        1
                    )

                    color_iter += 1

                    if color_iter >= len(colors):
                        color_iter = 0
    return img


def draw_skeleton(
        image: np.ndarray,
        humans: list,
        connect_indexes: list = CONNECT_INDEXES, color=(255, 0, 0), thickness=2,
        thr_hold=0.2):
    """
    Draw skeletons from `humans` list on image `image`

    Parameters
    ----------
    image : np.ndarray
        Image at which must be drawn skeletons
    humans : list
        List of Human class elements, which represent single human skeleton
    connect_indexes : list
        Pattern how skeletons must be connected,
        Default value is good enough for most cases
    color : tuple
        Color (R, G, B) - color of connected limbs
    thickness : int
        Thickness of drawn limb
    thr_hold : float
        Threshold for keypoints, by default equal to 0.2,
        If probability of points will be lower - this keypoint will have zero values,
        Work only if list of Human classes is input for this method

    Returns
    -------
    np.ndarray
        Image with skeletons on it

    """
    for indx in range(len(humans)):
        if isinstance(humans[indx], Human):
            human = humans[indx].to_list(th_hold=thr_hold)
        else:
            human = np.asarray(humans[indx]).reshape(-1, 3)

        for indx_limb in range(len(connect_indexes)):
            single_limb = connect_indexes[indx_limb]
            single_p1 = human[single_limb[0]]
            single_p2 = human[single_limb[1]]
            # if probability bigger than zero
            if single_p1[-1] > EPSILONE and single_p2[-1] > EPSILONE:
                p_1 = (int(single_p1[0]), int(single_p1[1]))
                p_2 = (int(single_p2[0]), int(single_p2[1]))
                cv2.line(image, p_1, p_2, color=color, thickness=thickness)
    return image
