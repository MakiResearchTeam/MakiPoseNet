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
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import cv2

from ..generators.pose_estimation.data_preparation import record_mp_pose_estimation_train_data
from pose_estimation.utils import CONNECT_INDEXES, scales_image_single_dim_keep_dims


class CocoPreparator:

    EPSILON = 1e-3
    KEYPOINTS_NUM = 24

    def __init__(self, 
            coco_annotations, 
            image_folder_path,
            max_people=8,
            min_image_size=512,
            is_use_strong_filter=True
    ):
        """
        Init objct that create tfrecords with data.
        NOTICE! Before use this class, you should enable eager execution mode in tensorflow,
        i.e. write `tf.compat.v1.enable_eager_execution()` at the beggining of the program

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
        is_use_strong_filter : bool
            If equal to True, then assume that the annotation has information as original coco json file,
            and to every annotation will be used filter to pick good one.
            If equal to False, then we assume that all annotation on images are good and further before save
            annotation, filter of bad annotation will be NOT used.
        """

        self._coco = COCO(coco_annotations)

        self._image_folder_path = image_folder_path
        
        self._max_people = max_people
        self._min_image_size = min_image_size
        self.__is_use_strong_filter = is_use_strong_filter
    
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
    
    def save_records(self, prefix, images_pr, stop_after_part=None, shuffle_data=True):
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
        shuffle_data : bool
            Shuffle data before start, if true

        """

        if self.__is_use_strong_filter:
            self.__create_records_with_strong_filter(prefix, images_pr, stop_after_part, shuffle_data)
        else:
            self.__create_records_wo_strong_filter(prefix, images_pr, stop_after_part, shuffle_data)

    def __create_records_wo_strong_filter(self, prefix, images_pr, stop_after_part=None, shuffle_data=True):
        ids_img = self._coco.getImgIds()
        if shuffle_data:
            ids_img = shuffle(ids_img)
        count_images = len(ids_img)
        part = count_images // images_pr
        iterator = tqdm(range(count_images))

        image_tensors = []
        image_masks = []
        keypoints_tensors = []
        keypoints_mask_tensors = []
        image_properties_tensors = []
        alpha_mask_tensors = []

        # Counter for saving parties
        counter = 0
        # Additional prefix for saved tfrecords
        counter_saved = 0

        for i in iterator:

            img_obj = self._coco.loadImgs(ids_img[i])[0]
            image = io.imread(os.path.join(self._image_folder_path, img_obj['file_name']))
            if img_obj.get('alpha_mask') is not None:
                alpha_mask = io.imread(os.path.join(self._image_folder_path, img_obj['alpha_mask']))
            else:
                # For this image - everything are HUMAN like, i.e. nothing will be changed
                alpha_mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
            annIds = self._coco.getAnnIds(imgIds=img_obj['id'], iscrowd=None)
            anns = self._coco.loadAnns(annIds)

            if len(anns) == 0:
                continue

            # Sort from the biggest person to the smallest one
            sorted_annot_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

            all_kp = []

            for people_n in sorted_annot_ids:
                single_person_data = anns[people_n]

                # (n_kp, 1, 3)
                all_kp_single = np.expand_dims(np.array(single_person_data['keypoints']).reshape(-1, 3), axis=1)
                # Create binary value for visibility
                all_kp_single[..., -1:] = (all_kp_single[..., -1:] > CocoPreparator.EPSILON).astype(np.float32, copy=False)
                all_kp.append(all_kp_single)

            if len(all_kp) == 0:
                continue

            all_kp = np.concatenate(all_kp, axis=1).astype(np.float32, copy=False)
            # Fill dimension n_people to maximum value according to self._max_people
            # By placing zeros in other additional dimensions
            not_enougth = self._max_people - all_kp.shape[1]
            if not_enougth > 0:
                zeros_arr = np.zeros([all_kp.shape[0], not_enougth, all_kp.shape[-1]]).astype(np.float32, copy=False)
                all_kp = np.concatenate([all_kp, zeros_arr], axis=1)

            if len(image.shape) != 3:
                # Assume that is gray-scale image, so convert it to rgb
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image_mask = np.ones((*image.shape[:2], 1)).astype(np.float32, copy=False)

            image, all_kp, image_mask, alpha_mask = self.__rescale_image(image, all_kp, image_mask, alpha_mask)

            keypoints_tensors.append(all_kp[..., :2].astype(np.float32, copy=False))
            keypoints_mask_tensors.append(all_kp[..., -1:].astype(np.float32, copy=False))

            image_tensors.append(image.astype(np.float32, copy=False))
            image_masks.append(image_mask)
            image_properties_tensors.append(np.array(image.shape).astype(np.float32, copy=False))
            alpha_mask_tensors.append(alpha_mask.astype(np.uint8, copy=False))
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
                    alpha_mask_tensors=alpha_mask_tensors,
                    prefix=prefix + f'_{counter_saved}',
                    dp_per_record=images_pr
                )

                image_tensors = []
                image_masks = []
                keypoints_tensors = []
                keypoints_mask_tensors = []
                image_properties_tensors = []
                alpha_mask_tensors = []
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
                alpha_mask_tensors=alpha_mask_tensors,
                prefix=prefix + f'_{counter_saved}',
                dp_per_record=images_pr
            )

        iterator.close()

    def __create_records_with_strong_filter(self, prefix, images_pr, stop_after_part=None, shuffle_data=True):
        ids_img = self._coco.getImgIds()
        if shuffle_data:
            ids_img = shuffle(ids_img)
        count_images = len(ids_img)
        part = count_images // images_pr
        iterator = tqdm(range(count_images))

        image_tensors = []
        image_masks = []
        keypoints_tensors = []
        keypoints_mask_tensors = []
        image_properties_tensors = []
        alpha_mask_tensors = []

        # Counter for saving parties
        counter = 0
        # Additional prefix for saved tfrecords
        counter_saved = 0

        for i in iterator:

            img_obj = self._coco.loadImgs(ids_img[i])[0]
            image = io.imread(os.path.join(self._image_folder_path, img_obj['file_name']))
            if img_obj.get('alpha_mask') is not None:
                alpha_mask = io.imread(os.path.join(self._image_folder_path, img_obj['alpha_mask']))
            else:
                # For this image - everything are HUMAN like, i.e. nothing will be changed
                alpha_mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
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
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32, copy=False))
                    continue
                # skip this person if parts number is too low or if
                # segmentation area is too small
                # Shape (n_keypoints, 1, 3)
                all_kp_single = self.take_default_skelet(single_person_data)

                if np.sum(all_kp_single[:, 0, -1]) < 5 or single_person_data["area"] < 32 * 32:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32, copy=False))
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
                    if dist < pc[2] * 0.3:
                        too_close = True
                        break

                if too_close:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    human_mask.append(self._coco.annToMask(single_person_data).astype(np.float32, copy=False))
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
                zeros_arr = np.zeros([all_kp.shape[0], not_enougth, all_kp.shape[-1]]).astype(np.float32, copy=False)
                all_kp = np.concatenate([all_kp, zeros_arr], axis=1)

            if len(image.shape) != 3:
                # Assume that is gray-scale image, so convert it to rgb
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if len(human_mask) > 0:
                image_mask = np.sum(human_mask, axis=0).astype(np.float32, copy=False)
                image_mask[image_mask > 0.0] = 1.0

                # Reverse
                image_mask = np.ones((*image.shape[:2], 1)).astype(np.float32, copy=False) - np.expand_dims(image_mask, axis=-1)
            else:
                image_mask = np.ones((*image.shape[:2], 1)).astype(np.float32, copy=False)

            image, all_kp, image_mask, alpha_mask = self.__rescale_image(image, all_kp, image_mask, alpha_mask)
            keypoints_tensors.append(all_kp[..., :2].astype(np.float32, copy=False))
            keypoints_mask_tensors.append(all_kp[..., -1:].astype(np.float32, copy=False))

            image_tensors.append(image.astype(np.float32, copy=False))
            image_masks.append(image_mask)
            image_properties_tensors.append(np.array(image.shape).astype(np.float32, copy=False))
            alpha_mask_tensors.append(alpha_mask.astype(np.uint8, copy=False))

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
                    alpha_mask_tensors=alpha_mask_tensors,
                    prefix=prefix + f'_{counter_saved}',
                    dp_per_record=images_pr
                )

                image_tensors = []
                image_masks = []
                keypoints_tensors = []
                keypoints_mask_tensors = []
                image_properties_tensors = []
                alpha_mask_tensors = []
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
                alpha_mask_tensors=alpha_mask_tensors,
                prefix=prefix + f'_{counter_saved}',
                dp_per_record=images_pr
            )

        iterator.close()

    def __rescale_image(self, image, keypoints, image_mask, alpha_mask):
        """
        Rescale image to minimum size, 
        if one of the dimension of the image (height and width) is smaller than `self._min_image_size`

        """

        xy_scales = scales_image_single_dim_keep_dims(
            image_size=image.shape[:-1],
            resize_to=self._min_image_size
        )
        h, w = image.shape[:2]

        new_w, new_h = (round(w * xy_scales[0]), round(h * xy_scales[1]))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        alpha_mask = np.expand_dims(cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_CUBIC), axis=-1)

        # In mask, cv2 drop last dimension because it equal 1
        image_mask = np.expand_dims(cv2.resize(image_mask, (new_w, new_h), interpolation=cv2.INTER_CUBIC), axis=-1)

        # Ignore dimension of visibility of the keypoints
        keypoints[..., :-1] *= np.array(xy_scales).astype(np.float32, copy=False)
        # Check bounds on Width dimension
        if new_w < self._min_image_size:
            # padding zeros to image and padding ones for image_mask
            padding_image = np.zeros((new_h, self._min_image_size, 3)).astype(np.float32, copy=False)
            padding_image[:, :new_w] = image

            padding_mask = np.ones((new_h, self._min_image_size, 1)).astype(np.float32, copy=False)
            padding_mask[:, :new_w] = image_mask

            padding_alpha_mask = np.ones((new_h, self._min_image_size, 1)).astype(np.uint8, copy=False)
            padding_alpha_mask *= np.min(alpha_mask)
            padding_alpha_mask[:, :new_w] = alpha_mask
            return padding_image, keypoints, padding_mask, padding_alpha_mask
        
        return image, keypoints, image_mask, alpha_mask

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
