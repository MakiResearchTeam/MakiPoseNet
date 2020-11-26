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

from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import skimage.io as io
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

from ..generators.pose_estimation.data_preparation import record_mp_pose_estimation_train_data


CONNECT_INDEXES =  [
    # head
    [1, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    # body
    # left
    [1, 7],
    [7, 9],
    [9, 11],
    [11, 22],
    [11, 23],
    # right
    [1, 6],
    [6, 8],
    [8, 10],
    [10, 20],
    [10, 21],
    # center
    [1, 0],
    [0, 12],
    [0, 13],
    # legs
    # left
    [13, 15],
    [15, 17],
    [17, 19],
    # right
    [12, 14],
    [14, 16],
    [16, 18]

]


class CocoPreparator:

    EPSILON = 1e-10

    def __init__(self, 
            coco_annotations, 
            image_folder_path,
            max_people=8,
            min_image_size=512, 
            criteria_throw=0.25
    ):
        """
        Parameters
        ----------
        coco_annotations : str
            Path to coco annotations file.
        image_folder_path : str
            Path to the folder containing images.
        max_people : int
            Images with more than `max_people` will be omitted.
        min_image_size : int
            Images with width or height less than `min_image_size` will be scaled to fulfill
            the criteria.
        criteria_throw : float
            Criteria relations of visible keypoints to invisible,
            This criteria is using in default method for checking number of keypoints on the image,
            All images with a lower relations will be thrown
        """
        # For saving records, enable eager execution
        tf.compat.v1.enable_eager_execution()

        self._coco = COCO(coco_annotations)

        self._image_folder_path = image_folder_path
        
        self._max_people = max_people
        self._min_image_size = min_image_size

        self.__criteria_throw = criteria_throw
    
    def show_annot(self, image_id, fig_size=[8, 8], color_limbs='b', color_skelet='b'):
        """
        Show image and skeletons on it according to `image_id`

        Parameters
        ----------
        image_id : int
            Id of image which should be shown with skeletons and keypoints
        fig_size : list
            Size of the matplotlib.pyplot figure,
            By default equal to [8, 8] which is enough for most cases
        color_limbs : str
            Color of points on certain limbs,
            'b' - blue,
            'r' - red,
            etc...
            For more details, check matplotlib color
        color_skelet : str
            Color of the skelet connection,
            'b' - blue,
            'r' - red,
            etc...
            For more details, check matplotlib color
        """

        plt.figure(figsize=fig_size)    
        annIds = self._coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        img = self._coco.loadImgs(image_id)[0]

        for z in range(len(anns)):
            # Method return shape (n_keypoints, 1, 3)
            all_kp = self.take_default_skelet(anns[z])[:, 0]

            for i in range(len(CONNECT_INDEXES)):
                single = CONNECT_INDEXES[i]
                p_1 = all_kp[single[0]]
                p_2 = all_kp[single[1]]
                if p_1[-1] > CocoPreparator.EPSILON and p_1[-1] > CocoPreparator.EPSILON and \
                   p_2[-1] > CocoPreparator.EPSILON and p_2[-1] > CocoPreparator.EPSILON:
                    plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=color_skelet)
                    plt.scatter([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=color_limbs)
        
        I = io.imread(os.path.join(self._image_folder_path, img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        plt.show()
    
    def save_records(self, prefix, images_pr, stop_after_part=None):
        """
        Parameters
        ----------
        prefix : str
            String containing path to save + prefix.
        images_pr : int
            Images per record. It is better to choose such a number that
            the records take about 200-300 mb of memory for the best pipeline
            performance.
        stop_after_part : int
            Generation of the tfrecords will be stopped after certain part,
            This parameters is usually used for creation a smaller tfrecord dataset
            (for example, for testing)
        """

        ids_img = self._coco.getImgIds()
        count_images = len(ids_img)
        part = count_images // images_pr
        iterator = tqdm(range(count_images))

        image_tensors = []
        image_masks = []
        keypoints_tensors = []
        keypoints_mask_tensors = []
        image_properties_tensors = []

        # Counter for saving parties
        counter = 0
        # Additional prefix for saved tfrecords
        counter_saved = 0

        for i in iterator:

            img_obj = self._coco.loadImgs(ids_img[i])[0]
            image = io.imread(os.path.join(self._image_folder_path, img_obj['file_name']))
            annIds = self._coco.getAnnIds(imgIds=img_obj['id'], iscrowd=None)
            anns = self._coco.loadAnns(annIds)

            if len(anns) == 0:
                continue

            # Sort from the biggest person to the smallest one
            sorted_annot_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

            all_kp = []
            human_mask = []

            # To keep previous center point of human position
            prev_center = []

            for people_n in sorted_annot_ids:
                single_person_data = anns[people_n]

                if single_person_data["iscrowd"] or len(all_kp) >= self._max_people:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32))
                    continue
                # skip this person if parts number is too low or if
                # segmentation area is too small
                # Shape (n_keypoints, 1, 3)
                all_kp_single = self.take_default_skelet(single_person_data)

                if np.sum(all_kp_single[:, 0, -1]) < 5 or single_person_data["area"] < 32 * 32:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32))
                    continue

                person_center = [
                    single_person_data["bbox"][0] + single_person_data["bbox"][2] / 2,
                    single_person_data["bbox"][1] + single_person_data["bbox"][3] / 2
                ]

                # skip this person if the distance to existing person is too small

                too_close = False
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2]*0.3:
                        too_close = True
                        break

                if too_close:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32))
                    continue

                prev_center.append(
                    np.append(
                        person_center,
                        max(single_person_data["bbox"][2], single_person_data["bbox"][3])
                    )
                )

                all_kp.append(all_kp_single)

            if len(all_kp) == 0:
                continue

            all_kp = np.concatenate(all_kp, axis=1)
            # Fill dimension n_people to maximum value according to self._max_people 
            # By placing zeros in other additional dimensions
            not_enougth = self._max_people - all_kp.shape[1]
            if not_enougth > 0:
                zeros_arr = np.zeros([all_kp.shape[0], not_enougth, all_kp.shape[-1]]).astype(np.float32)
                all_kp = np.concatenate([all_kp, zeros_arr], axis=1)

            if len(image.shape) != 3:
                # Assume that is gray-scale image, so convert it to rgb
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if len(human_mask) > 0:
                image_mask = np.sum(human_mask, axis=0).astype(np.float32)
                image_mask[image_mask > 0.0] = 1.0

                # Reverse
                image_mask = np.ones((*image.shape[:2], 1)).astype(np.float32) - np.expand_dims(image_mask, axis=-1)
            else:
                image_mask = np.ones((*image.shape[:2], 1)).astype(np.float32)

            image, all_kp, image_mask = self.__rescale_image(image, all_kp, image_mask)

            keypoints_tensors.append(all_kp[..., :2].astype(np.float32))
            keypoints_mask_tensors.append(all_kp[..., -1:].astype(np.float32))
            
            image_tensors.append(image.astype(np.float32))
            image_masks.append(image_mask)
            image_properties_tensors.append(np.array(image.shape).astype(np.float32))
            
            counter += 1
            
            # For large datasets, it's better to save them by parties
            if counter == part:
                print('Save part of the tfrecords...')
                record_mp_pose_estimation_train_data(
                    image_tensors=image_tensors,
                    image_masks=image_masks,
                    keypoints_tensors=keypoints_tensors,
                    keypoints_mask_tensors=keypoints_mask_tensors,
                    image_properties_tensors=image_properties_tensors,
                    prefix=prefix + f'_{counter_saved}',
                    dp_per_record=images_pr
                )

                image_tensors = []
                image_masks = []
                keypoints_tensors = []
                keypoints_mask_tensors = []
                image_properties_tensors = []
                counter = 0
                counter_saved += 1

            if stop_after_part is not None and counter_saved == stop_after_part:
                break

        # Save remaining data
        if counter > 0:
            print('Save part of the tfrecords...')
            record_mp_pose_estimation_train_data(
                image_tensors=image_tensors,
                image_masks=image_masks,
                keypoints_tensors=keypoints_tensors,
                keypoints_mask_tensors=keypoints_mask_tensors,
                image_properties_tensors=image_properties_tensors,
                prefix=prefix + f'_{counter_saved}',
                dp_per_record=images_pr
            )

        iterator.close()

    def __rescale_image(self, image, keypoints, image_mask):
        """
        Rescale image to minimum size, 
        if one of the dimension of the image (height and width) is smaller than `self._min_image_size`

        """
        if image.shape[0] < self._min_image_size or image.shape[1] < self._min_image_size:
            h, w = image.shape[:2]
            min_dim = min(h, w)
            scale = self._min_image_size / min_dim
            w, h = round(w * scale), round(h * scale)
            image = cv2.resize(image, (w, h))
            image_mask = np.expand_dims(cv2.resize(image_mask, (w, h)), axis=-1)
            # Ignore dimension of visibility of the keypoints 
            keypoints[..., :-1] *= scale
        
        return image, keypoints, image_mask

    @staticmethod
    def take_default_skelet(single_human_anns):
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
            foot_one = np.zeros(3).astype(np.float32)
        # Two
        if single_anns_foot[3][-1] > CocoPreparator.EPSILON and single_anns_foot[4][-1] > CocoPreparator.EPSILON:
            foot_two = (single_anns_foot[3] + single_anns_foot[4]) / 2.0
            # Set visibility to True (i.e. 1.0)
            foot_two[-1] = 1.0
        else:
            foot_two = np.zeros(3).astype(np.float32)

        # Check neck
        if single_anns[5][-1] > CocoPreparator.EPSILON and single_anns[6][-1] > CocoPreparator.EPSILON:
            chest_p = (single_anns[5] + single_anns[6]) / 2.0

            face_p = np.zeros(3).astype(np.float32)
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
                neck = np.zeros(3).astype(np.float32)
        else:
            neck = np.zeros(3).astype(np.float32)

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
            avg_body_p = np.zeros(3).astype(np.float32)
            div = 0
            for i in range(len(s_p_imp)):
                if s_p_imp[i][-1] > CocoPreparator.EPSILON and s_p_imp[i][-1] > CocoPreparator.EPSILON:
                    div += 1
                    avg_body_p += s_p_imp[i]

            avg_body_p = avg_body_p / div
            avg_body_p[-1] = 1.0
        else:
            # Otherwise the point is not visible
            avg_body_p = np.zeros(3).astype(np.float32)

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
        all_kp_single[..., -1:] = (all_kp_single[..., -1:] > CocoPreparator.EPSILON).astype(np.float32)
        # all_kp - shape (24, 3) ---> (24, 1, 3)
        all_kp_single = np.expand_dims(all_kp_single, axis=1)

        return all_kp_single
