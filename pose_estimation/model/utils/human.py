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

from .constants import NUMBER_OF_KEYPOINTS
import numpy as np


class Human:
    """
    Store keypoints of the single human

    """
    __slots__ = ('body_parts', 'score')

    def __init__(self):
        """
        Init class to store keypoints of the single human

        """
        self.body_parts = {}
        self.score = 0.0

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def to_list(self, th_hold=0.2) -> list:
        """
        Transform keypoints stored in this class to list

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        list
            List with lenght NK * 3, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for visibility of the points
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        list_data = []
        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is None or take_single.score < th_hold:
                list_data += [0.0, 0.0, 0.0]
            else:
                list_data += [
                    self.body_parts[i].x,
                    self.body_parts[i].y,
                    self.body_parts[i].score,
                ]

        return list_data

    def to_dict(self, th_hold=0.2, skip_not_visible=False, key_as_int=False) -> dict:
        """
        Transform keypoints stored in this class to dict

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        skip_not_visible : bool
            If equal to True, then values with low probability (or invisible)
            Will be skipped from final dict

        Returns
        -------
        dict
            Dict of the keypoints,
            { NumKeypoints:   [x_coord, y_coord, score],
              NumKeypoints_1: [x_coord, y_coord, score],
              ..........................................
            }
            Where NumKeypoints, NumKeypoints_1 ... are string values responsible for index of the keypoint,
            x_coord - coordinate of the keypoint on X axis
            y_coord - coordinate of the keypoint on Y axis
            score - confidence of the neural network
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        dict_data = {}
        if key_as_int:
            key_tr = lambda x: int(x)
        else:
            key_tr = lambda x: str(x)

        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is not None and take_single.score >= th_hold:
                dict_data.update({
                    key_tr(i): [take_single.x, take_single.y, take_single.score]
                })
            elif not skip_not_visible:
                dict_data.update({
                    key_tr(i): [0.0, 0.0, 0.0]
                })

        return dict_data

    def to_np(self, th_hold=0.2):
        """
        Transform keypoints stored in this class to numpy array with shape (N, 3),
        Where N - number of points

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        np.ndarray
            Array of keypoints with shape (N, 3),
            Where N - number of points

        """
        list_points = self.to_list(th_hold=th_hold)
        # (N, 3)
        return np.array(list_points, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def from_array(skeleton_array):
        """
        Take points from `skeleton_array` and create Human class with this points

        Parameters
        ----------
        skeleton_array : np.ndarray or list
            Array of input points
            NOTICE! Input array must be with shape (N, 3) (N - number of points)

        Returns
        -------
        Human
            Created Human class with points in `skeleton_np`

        """
        human_class = Human()
        human_id = 0
        sum_probs = 0.0

        for part_idx in range(len(skeleton_array)):
            human_class.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(skeleton_array[part_idx][0]),
                float(skeleton_array[part_idx][1]),
                float(skeleton_array[part_idx][-1])
            )
            sum_probs += float(skeleton_array[part_idx][-1])

        human_class.score = sum_probs / len(skeleton_array)
        return human_class

    @staticmethod
    def from_dict(skeleton_dict):
        """
        Take points from `skeleton_dict` and create Human class with this points

        Parameters
        ----------
        skeleton_dict : dict
            Dict of input points
            Example:
            {
                0: [22.0, 23.0, 1.0],
                1: [10, 20, 0.2],
                ....
            }

        Returns
        -------
        Human
            Created Human class with points in `skeleton_dict`

        """
        human_class = Human()
        human_id = 0
        sum_probs = 0.0
        human_class.score = 0.0

        for part_idx, v_arr in skeleton_dict.items():
            human_class.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(v_arr[0]),
                float(v_arr[1]),
                float(v_arr[-1])
            )
            sum_probs += float(v_arr[-1])
        if len(skeleton_dict) >= 1:
            human_class.score = sum_probs / len(skeleton_dict)
        return human_class

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    Store single keypoints with certain coordinates and score

    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        """
        Init

        Parameters
        ----------
        uidx : str
            String stored number of the human and number of this keypoint
        part_idx :
        x : float
            Coordinate of the keypoint at the x-axis
        y : float
            Coordinate of the keypoint at the y-axis
        score : float
            Confidence score from neural network
        """
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()
