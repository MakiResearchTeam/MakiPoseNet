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
from .human import Human, BodyPart
import itertools

try:
    from .pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess.')
    exit(-1)


def estimate_paf(peaks, indices, paf_mat) -> list:
    """
    Estimate paff by using heatmap and peaks

    Parameters
    ----------
    peaks : np.ndarray
        [N], value of peak
    indices : np.ndarray
        [N, 3], first 2 dimensions - yx coord, last dimension - keypoint class
    paf_mat : np.ndarray
        Numpy array of the PAF (Party affinity fields) which is usually prediction of the network

    Returns
    -------
    list
        List of the Human which contains body keypoints
    """
    pafprocess.process_paf(peaks, indices, paf_mat)
    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human()
        is_added = False

        for part_idx in range(NUMBER_OF_KEYPOINTS):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)),
                float(pafprocess.get_part_y(c_idx)),
                pafprocess.get_part_score(c_idx)
            )
        # if at least one keypoint was visible add `human` to all humans
        # Otherwise skip
        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans


def merge_similar_skelets(humans: list, th_hold_x=0.5, th_hold_y=0.5) -> list:
    """
    Merge similar skeletons into one skelet

    Parameters
    ----------
    humans : list
        List of the predicted skelets from `estimate_paf` script
    th_hold_x : float
        Threshold from what value do we count keypoints similar by axis X,
        By default equal to 0.04
    th_hold_y : float
        Threshold from what value do we count keypoints similar by axis Y,
        By default equal to 0.04

    Returns
    -------
    list
        List of the predicted people.
        Single element is a Human class
    """
    # Store { num_human: Human_class }
    humans_dict = dict([(str(i), humans[i]) for i in range(len(humans))])

    # Merge all skeletons
    # While there is no any need in merge operation
    while True:
        is_merge = False

        # Combinations of all humans that predict Neural Network
        for h1, h2 in itertools.combinations(list(range(len(humans_dict))), 2):
            # If any of the human were deleted, just skip it
            if humans_dict.get(str(h1)) is None or humans_dict.get(str(h2)) is None:
                continue
            # Check all keypoints on distance
            for c1, c2 in itertools.product(humans_dict[str(h1)].body_parts, humans_dict[str(h2)].body_parts):
                single_keypoints_1 = humans[h1].body_parts[c1]
                single_keypoints_2 = humans[h2].body_parts[c2]

                # If any keypoints very close to other, just merge all skeleton and quit from this loop
                if (abs(single_keypoints_1.x - single_keypoints_2.x) < th_hold_x and
                    abs(single_keypoints_1.y - single_keypoints_2.y) < th_hold_y
                ):

                    is_merge = True
                    humans_dict[str(h1)].body_parts.update(humans[h2].body_parts)
                    humans_dict.pop(str(h2))
                    break

        # Quit if there are no any need in merge operation, i.e. it was not used
        if not is_merge:
            break

    return list(humans_dict.values())
