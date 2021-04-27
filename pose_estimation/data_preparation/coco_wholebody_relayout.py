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

import numpy as np
from tqdm import tqdm
import copy
import json

from pose_estimation.data_preparation.coco_preparator_api import CocoPreparator

# Annotations in JSON
ANNOTATIONS = 'annotations'
IMAGES = 'images'
BBOX = 'bbox'
SEGMENTATION = 'segmentation'
KEYPOINTS = 'keypoints'
AREA = 'area'
# Stored in the annotations
IMAGE_ID = 'image_id'
# Id in the image
ID = 'id'

# Images in JSON
HEIGHT = 'height'
WIDTH = 'width'


EPS = 1e-3


class KpAndIndexStruct:
    """
    Store kp (x, y and visibility value)
    and index in array of this keypoint

    """
    def __init__(self, kp, indx):
        """

        Parameters
        ----------
        kp : np.ndarray
            (x, y, v)
        indx : int
            Indx of this keypoints in final array of all keypoints

        """
        self.kp = kp
        self.indx = indx


class CocoWholeBodyRelayout:
    """
    Map CocoWHoleBody to other skeleton
    By default repo use only keypoints field in JSON data
    With this class, these points cna be modified/added/deleted

    Notice! First 18 keypoints always same
    For more detail refer to `setup_taken_points_from_foots`, `setup_taken_points_from_left_hand`
    and `setup_taken_points_from_right_hand` docs of methods

    """

    def __init__(self, ann_file_path: str):
        """
        Relayout original annotation to suitable one for further purposes

        Parameters
        ----------
        ann_file_path : str
            Path to the origin annotation file
            Example: /home/user_1/annot.json

        """
        self._ann_file_path = ann_file_path
        with open(ann_file_path, 'r') as fp:
            self._cocoGt_json = json.load(fp)
        self._kp_from_keypoints = []
        self._kp_from_foots = []
        self._kp_from_hands = []

    def setup_taken_points_from_foots(self, kp_indx):
        """
        Add additional points into skeleton
        Notice! Default skeleton have 18 keypoints (kp),
        So, `kp_indx` mean that `18 + kp_indx` - is your indx at final array (so its start from zero)
        Don't forget about that!

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        pass

    def setup_taken_points_from_left_hand(self, kp_take, kp_indx):
        """
        Add additional points into skeleton
        Notice! Default skeleton have 18 keypoints (kp),
        So, `kp_indx` mean that `18 + kp_indx` - is your indx at final array (so its start from zero)
        Don't forget about that!

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        pass

    def setup_taken_points_from_right_hand(self, kp_take, kp_indx):
        """
        Add additional points into skeleton
        Notice! Default skeleton have 18 keypoints (kp),
        So, `kp_indx` mean that `18 + kp_indx` - is your indx at final array (so its start from zero)
        Don't forget about that!

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        pass

    def relayout(self, path_to_save, limit_number):
        """
        Relayout original annotation to suitable one for further purposes

        Parameters
        ----------
        path_to_save : str
            Path where need save a relayout annotation file
        limit_number : int
            Limit number of loaded annotation,
            If equal to None then all annotations will be loaded

        """

        # Store: (image_id, image_info)
        dict_id_by_image_info = dict([(elem[ID], elem) for elem in self._cocoGt_json[IMAGES]])
        # Store: (image_id, bool)
        used_ids = dict([(elem[ID], False) for elem in self._cocoGt_json[IMAGES]])

        Maki_cocoGt_json = copy.deepcopy(self._cocoGt_json)
        # Clear information about annotations and images
        # In next for loop, we write new information
        Maki_cocoGt_json[ANNOTATIONS] = []
        Maki_cocoGt_json[IMAGES] = []

        if limit_number is None:
            iterator = tqdm(range(len(self._cocoGt_json[ANNOTATIONS])))

        elif type(limit_number) == int:
            iterator = tqdm(range(min(limit_number, len(self._cocoGt_json[ANNOTATIONS]))))

        else:
            raise TypeError(f'limit_number should have type int, but it has {type(limit_number)} '
                            f'and value {limit_number}')

        for i in iterator:
            single_anns = self._cocoGt_json[ANNOTATIONS][i]
            new_keypoints = np.array(single_anns[KEYPOINTS]).reshape(-1, 3)
            if new_keypoints.shape[0] != CocoPreparator.KEYPOINTS_NUM:
                new_keypoints = self.take_default_skelet(single_anns)
            image_annot = self.find_image_annot(self._cocoGt_json, single_anns[IMAGE_ID])
            if image_annot is None:
                raise ModuleNotFoundError(f'Image id: {single_anns[IMAGE_ID]} was not found.')

            new_segmentation = single_anns.get(SEGMENTATION)
            # There is some garbage that stored in segmentation dict
            # Just skip it
            # TODO: Do something with this images
            if new_segmentation is not None and type(new_segmentation) == dict:
                continue

            # Fill our annotation with new information
            single_anns[KEYPOINTS] = new_keypoints.reshape(-1).astype(np.float32, copy=False).tolist()
            Maki_cocoGt_json[ANNOTATIONS].append(single_anns)

            # Write img ids which we process
            if not used_ids[single_anns[IMAGE_ID]]:
                Maki_cocoGt_json[IMAGES].append(dict_id_by_image_info[single_anns[IMAGE_ID]])
                used_ids[single_anns[IMAGE_ID]] = True

        iterator.close()
        with open(path_to_save, 'w') as fp:
            json.dump(Maki_cocoGt_json, fp)

    def find_image_annot(self, cocoGt_json: dict, img_id: int) -> dict:
        """
        Return annotation from `cocoGt_json` annotation according to `img_id`

        """
        for single_annot in cocoGt_json[IMAGES]:
            if single_annot[ID] == img_id:
                return single_annot

        return None

    def take_default_skelet(self, single_human_anns):
        """
        Take default skelet with 24 keypoints for full body

        Returns
        -------
        np.ndarray
            Array of the keypoints with shape - (24, 1, 3)

        """
        single_anns = np.array(single_human_anns['keypoints']).reshape(17, 3)
        single_anns_hand_left = np.array(single_human_anns['lefthand_kpts']).reshape(21, 3)
        single_anns_hand_right = np.array(single_human_anns['righthand_kpts']).reshape(21, 3)
        single_anns_foot = np.array(single_human_anns['foot_kpts']).reshape(6, 3)

        # Check visibility of foots
        # One
        if single_anns_foot[0][-1] > CocoPreparator.EPSILON and single_anns_foot[1][-1] > CocoPreparator.EPSILON:
            foot_one = (single_anns_foot[0] + single_anns_foot[1]) / 2.0
            # Set visibility to True (i.e. 1.0)
            foot_one[-1] = 1.0
        else:
            foot_one = np.zeros(3).astype(np.float32, copy=False)
        # Two
        if single_anns_foot[3][-1] > CocoPreparator.EPSILON and single_anns_foot[4][-1] > CocoPreparator.EPSILON:
            foot_two = (single_anns_foot[3] + single_anns_foot[4]) / 2.0
            # Set visibility to True (i.e. 1.0)
            foot_two[-1] = 1.0
        else:
            foot_two = np.zeros(3).astype(np.float32, copy=False)

        # Check neck
        if single_anns[5][-1] > CocoPreparator.EPSILON and single_anns[6][-1] > CocoPreparator.EPSILON:
            chest_p = (single_anns[5] + single_anns[6]) / 2.0

            face_p = np.zeros(3).astype(np.float32, copy=False)
            # Calculate average points position on the face using know points on it
            div = 0
            for i in range(len(single_anns[:5])):
                if single_anns[i][-1] > CocoPreparator.EPSILON and single_anns[i][-1] > CocoPreparator.EPSILON:
                    div += 1
                    face_p += single_anns[i]

            # Check whether there points on the face, ignore if there is 3 or less points
            if div > 3:
                face_p = face_p / div

                # Calculate point which is on vector from points `chest_p` to `face_p`,
                # We take points at the end of this vector on 1/3 of its length
                neck = (face_p - chest_p) / 3.0 + chest_p
                # Set visibility to True (i.e. 1.0)
                neck[-1] = 1.0
            else:
                neck = np.zeros(3).astype(np.float32, copy=False)
        else:
            neck = np.zeros(3).astype(np.float32, copy=False)

        # Middle points of the body
        # Calculate avg points between 4 body points
        s_p_imp = np.stack([
            single_anns[5],
            single_anns[6],
            single_anns[11],
            single_anns[12]
        ])
        # Making binary values on visibility dimension to calculate visible points
        s_p_imp[..., -1:] = s_p_imp[..., -1:] > CocoPreparator.EPSILON
        # For middle points, we must have more than 2 visible points for its calculation
        number_vis_points = np.sum(s_p_imp[..., -1:])

        if number_vis_points >= 3:
            # The point is visible
            avg_body_p = np.zeros(3).astype(np.float32, copy=False)
            div = 0
            for i in range(len(s_p_imp)):
                if s_p_imp[i][-1] > CocoPreparator.EPSILON and s_p_imp[i][-1] > CocoPreparator.EPSILON:
                    div += 1
                    avg_body_p += s_p_imp[i]

            avg_body_p = avg_body_p / div
            avg_body_p[-1] = 1.0
        else:
            # Otherwise the point is not visible
            avg_body_p = np.zeros(3).astype(np.float32, copy=False)

        all_kp_single = np.stack([
            avg_body_p,
            neck,
            *list(single_anns[1:5]),
            *list(single_anns[5:]),
            # foot
            foot_one,
            foot_two,
            # hands
            single_anns_hand_left[4],
            single_anns_hand_left[-1],
            single_anns_hand_right[4],
            single_anns_hand_right[-1],
        ])

        # Create bool mask for visibility of keypoints
        all_kp_single[..., -1:] = (all_kp_single[..., -1:] > CocoPreparator.EPSILON).astype(np.float32, copy=False)
        # all_kp - shape (24, 3) ---> (24, 1, 3)
        all_kp_single = np.expand_dims(all_kp_single, axis=1)

        return all_kp_single
