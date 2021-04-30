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


class CocoWholeBodyRelayout:
    """
    Map CocoWHoleBody to other skeleton
    By default repo use only keypoints field in JSON data
    With this class, these points cna be modified/added/deleted

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
        # dict
        # indx_kp_set : indx_kp_from_whole_body
        self._kp_from_keypoints = dict()
        self._kp_from_foots = dict()
        self._kp_from_hands_left = dict()
        self._kp_from_hands_right = dict()
        # neck params
        self._set_is_calculate_neck = False
        self._neck_indx = None
        # center body params
        self._set_is_calculate_center_body_kp = False
        self._center_body_indx = None
        # mid finger params
        self._set_is_calculate_mid_foot_kp = False
        self._foot_mid_indx = None

    def set_calculate_neck(self, kp_indx_set=0):
        """
        Setup neck point

        Parameters
        ----------
        is_calculate : bool
            If True, then kp will be calculated
        kp_indx_set : int
            Index of neck point in overall array of kp

        """
        self._set_is_calculate_neck = True
        self._neck_indx = kp_indx_set

    def set_calculate_mid_foot_finger(self, kp_indx_set: list):
        """
        Setup mid foot finger

        Parameters
        ----------
        kp_indx_set : list
            List of indx of kp in overall array,
            Order: left indx first, right last

        """
        self._set_is_calculate_mid_foot_kp = True
        self._foot_mid_indx = kp_indx_set

    def set_calculate_center_body(self, kp_indx_set: int):
        """
        Setup center body point

        Parameters
        ----------
        kp_indx_set : int
            Indx of kp in overall array

        """
        self._set_is_calculate_center_body_kp = True
        self._center_body_indx = kp_indx_set

    def setup_taken_points_from_foots(self, kp_indx_stored: list, kp_indx_set: list):
        """
        Map kp from foot to final array of kp
        Example:
            kp_indx_stored = [1, 2, 10]
            kp_kp_indx_set = [10, 11, 12]
        Then kp from foot array with indxes: 1, 2, 10 will be taken and assign in overall array
        With indxes 10, 11, 12

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        if len(kp_indx_stored) != len(kp_indx_set):
            raise ValueError('`kp_indx_stored` and `kp_indx_set` must be list of same lenght')

        for s_indx_stored, s_indx_set in zip(kp_indx_stored, kp_indx_set):
            self._kp_from_foots[str(s_indx_set)] = s_indx_stored

    def setup_taken_points_from_left_hand(self, kp_indx_stored: list, kp_indx_set: list):
        """
        Map kp from left hand to final array of kp
        Example:
            kp_indx_stored = [1, 2, 10]
            kp_kp_indx_set = [10, 11, 12]
        Then kp from foot array with indxes: 1, 2, 10 will be taken and assign in overall array
        With indxes 10, 11, 12

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        if len(kp_indx_stored) != len(kp_indx_set):
            raise ValueError('`kp_indx_stored` and `kp_indx_set` must be list of same lenght')

        for s_indx_stored, s_indx_set in zip(kp_indx_stored, kp_indx_set):
            self._kp_from_hands_left[str(s_indx_set)] = s_indx_stored

    def setup_taken_points_from_right_hand(self, kp_indx_stored: list, kp_indx_set: list):
        """
        Map kp from right hand to final array of kp
        Example:
            kp_indx_stored = [1, 2, 10]
            kp_kp_indx_set = [10, 11, 12]
        Then kp from foot array with indxes: 1, 2, 10 will be taken and assign in overall array
        With indxes 10, 11, 12

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        if len(kp_indx_stored) != len(kp_indx_set):
            raise ValueError('`kp_indx_stored` and `kp_indx_set` must be list of same lenght')

        for s_indx_stored, s_indx_set in zip(kp_indx_stored, kp_indx_set):
            self._kp_from_hands_right[str(s_indx_set)] = s_indx_stored

    def setup_taken_points_from_keypoints(self, kp_indx_stored: list, kp_indx_set: list):
        """
        Map kp from keypoints to final array of kp
        Example:
            kp_indx_stored = [1, 2, 10]
            kp_kp_indx_set = [10, 11, 12]
        Then kp from foot array with indxes: 1, 2, 10 will be taken and assign in overall array
        With indxes 10, 11, 12

        Parameters
        ----------
        kp_take : int
            Index in WholtBody json
        kp_indx : int
            Index where will be assign this taken keypoint

        """
        if len(kp_indx_stored) != len(kp_indx_set):
            raise ValueError('`kp_indx_stored` and `kp_indx_set` must be list of same lenght')

        for s_indx_stored, s_indx_set in zip(kp_indx_stored, kp_indx_set):
            self._kp_from_keypoints[str(s_indx_set)] = s_indx_stored

    def get_current_setup(self) -> bool:
        """
        print current array of setup skeleton
        and checks for correctness

        Returns
        -------
        bool
            If true, then final skeleton in good and can be created
            otherwise, there is somewhere error in created skeleton
        """
        is_all_good = True
        # check same indx in all arrays and drop error if appears
        # foot
        check_indx = list(self._kp_from_foots.keys())
        all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_hands_left.keys()) + \
                       list(self._kp_from_keypoints.keys())
        for single_elem in check_indx:
            if single_elem in all_other_kp:
                raise ValueError(f"Index {single_elem} in foot array appear in other parts of array.")

        # left hand
        check_indx = list(self._kp_from_hands_left.keys())
        all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_foots.keys()) + \
                       list(self._kp_from_keypoints.keys())
        for single_elem in check_indx:
            if single_elem in all_other_kp:
                raise ValueError(f"Index {single_elem} in left hand array appear in other parts of array.")
        # right hand
        check_indx = list(self._kp_from_hands_right.keys())
        all_other_kp = list(self._kp_from_foots.keys()) + list(self._kp_from_hands_left.keys()) + \
                       list(self._kp_from_keypoints.keys())
        for single_elem in check_indx:
            if single_elem in all_other_kp:
                raise ValueError(f"Index {single_elem} in right hand array appear in other parts of array.")
        # keypoints
        check_indx = list(self._kp_from_keypoints.keys())
        all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_hands_left.keys()) + \
                       list(self._kp_from_foots.keys())
        for single_elem in check_indx:
            if single_elem in all_other_kp:
                raise ValueError(f"Index {single_elem} in keypoints array appear in other parts of array.")
        # check extra points
        # mid foot
        if self._set_is_calculate_mid_foot_kp:
            check_indx = self._foot_mid_indx
            all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_hands_left.keys()) + \
                           list(self._kp_from_foots.keys()) + list(self._kp_from_keypoints.keys()) + \
                           [self._neck_indx] + [self._center_body_indx]
            for single_elem in check_indx:
                if single_elem in all_other_kp:
                    raise ValueError(f"Index {single_elem} of mid foot array appear in other parts of array.")
        # center body
        if self._set_is_calculate_center_body_kp:
            check_indx = [self._center_body_indx]
            all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_hands_left.keys()) + \
                           list(self._kp_from_foots.keys()) + list(self._kp_from_keypoints.keys()) + \
                           [self._neck_indx] + self._foot_mid_indx
            for single_elem in check_indx:
                if single_elem in all_other_kp:
                    raise ValueError(f"Index {single_elem} of center body array appear in other parts of array.")
        # neck
        if self._set_is_calculate_neck:
            check_indx = [self._neck_indx]
            all_other_kp = list(self._kp_from_hands_right.keys()) + list(self._kp_from_hands_left.keys()) + \
                           list(self._kp_from_foots.keys()) + list(self._kp_from_keypoints.keys()) + \
                           [self._center_body_indx] + self._foot_mid_indx
            for single_elem in check_indx:
                if single_elem in all_other_kp:
                    raise ValueError(f"Index {single_elem} of neck array appear in other parts of array.")

        # Check order of kp
        all_points_k = list(self._kp_from_keypoints.keys()) + list(self._kp_from_hands_right.keys()) + \
                       list(self._kp_from_hands_left.keys()) + list(self._kp_from_foots.keys())

        if self._set_is_calculate_neck:
            all_points_k += [str(self._neck_indx)]

        if self._set_is_calculate_center_body_kp:
            all_points_k += [str(self._center_body_indx)]

        if self._set_is_calculate_mid_foot_kp:
            all_points_k += list(map(lambda x: str(x), self._foot_mid_indx))

        all_points_k = set(all_points_k)
        print(f'Number of keypoints: {len(all_points_k)}')
        print('Check index of skeletons...')
        for i in range(len(all_points_k)):
            if not str(i) in all_points_k:
                print("Points with index: ", i, " not present in overall array, \nDid you mess some keypoint?")
                is_all_good = False
        print('Check each part...')
        # print in order keypoints and print from which parts it is
        for i in range(len(all_points_k)):
            # take kp from single dict and print info
            # check extra points
            # mid foot
            if self._set_is_calculate_mid_foot_kp and str(i) in self._foot_mid_indx:
                print(f'index: {i} - mid foot kp')
                continue
            # center body
            if self._set_is_calculate_center_body_kp and self._center_body_indx == i:
                print(f'index: {self._center_body_indx} - center body')
                continue
            # neck
            if self._set_is_calculate_neck and self._neck_indx == i:
                print(f'index: {self._neck_indx} - neck')
                continue

            # check foot
            taken_kp = self._kp_from_foots.get(str(i))
            if taken_kp is not None:
                print(f'index: {i} taken from foots with index: {taken_kp}')
                continue
            # check left hand
            taken_kp = self._kp_from_hands_left.get(str(i))
            if taken_kp is not None:
                print(f'index: {i} taken from left hand with index: {taken_kp}')
                continue
            # check right hand
            taken_kp = self._kp_from_foots.get(str(i))
            if taken_kp is not None:
                print(f'index: {i} taken from right hand with index: {taken_kp}')
                continue
            # check keypoints
            taken_kp = self._kp_from_keypoints.get(str(i))
            if taken_kp is not None:
                print(f'index: {i} taken from keypoints with index: {taken_kp}')
                continue
        return is_all_good

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
        # Check final skeleton
        if not self.get_current_setup():
            print("Something wrong in created skeleton.\nPlease check your config")
            return
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
        all_points_k = list(self._kp_from_keypoints.keys()) + list(self._kp_from_hands_right.keys()) + \
                       list(self._kp_from_hands_left.keys()) + list(self._kp_from_foots.keys())

        if self._set_is_calculate_neck:
            all_points_k += [self._neck_indx]

        if self._set_is_calculate_center_body_kp:
            all_points_k += [self._center_body_indx]

        if self._set_is_calculate_mid_foot_kp:
            all_points_k += self._foot_mid_indx

        all_kp_single = np.zeros((len(all_points_k), 3)).astype(np.float32)

        single_anns = np.array(single_human_anns['keypoints']).reshape(17, 3)
        single_anns_hand_left = np.array(single_human_anns['lefthand_kpts']).reshape(21, 3)
        single_anns_hand_right = np.array(single_human_anns['righthand_kpts']).reshape(21, 3)
        single_anns_foot = np.array(single_human_anns['foot_kpts']).reshape(6, 3)
        # Take kp from singlee annot
        # keypoints
        for k, v in self._kp_from_keypoints.items():
            all_kp_single[int(k)] = single_anns[v]

        # left hand
        for k, v in self._kp_from_hands_left.items():
            all_kp_single[int(k)] = single_anns_hand_left[v]

        # right hand
        for k, v in self._kp_from_hands_right.items():
            all_kp_single[int(k)] = single_anns_hand_right[v]

        # foot
        for k, v in self._kp_from_foots.items():
            all_kp_single[int(k)] = single_anns_foot[v]
        # Some extra points
        if self._set_is_calculate_mid_foot_kp:
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
            all_kp_single[self._foot_mid_indx[0]] = foot_one # todo: check right and left sides
            all_kp_single[self._foot_mid_indx[1]] = foot_two

        if self._set_is_calculate_neck:
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
            all_kp_single[self._neck_indx] = neck

        if self._set_is_calculate_center_body_kp:
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
            all_kp_single[self._center_body_indx] = avg_body_p
        # todo: remove
        """
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
        """
        # Create bool mask for visibility of keypoints
        all_kp_single[..., -1:] = (all_kp_single[..., -1:] > CocoPreparator.EPSILON).astype(np.float32, copy=False)
        return all_kp_single
