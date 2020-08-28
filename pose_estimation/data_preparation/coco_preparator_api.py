from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import skimage.io as io
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
            criteria_throw=0.4
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
    
    def show_annot(self, image_id, fig_size=[8, 8]):
        """
        Show image and skeletons on it according to `image_id`

        Parameters
        ----------
        image_id : int
            Id of image which should be shown with skeletons and keypoints
        fig_size : list
            Size of the matplotlib.pyplot figure,
            By default equal to [8, 8] which is enough for most cases
        """

        plt.figure(figsize=fig_size)    
        annIds = self._coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        img = self._coco.loadImgs(image_id)[0]

        for z in range(len(anns)):
            all_kp = self.__take_default_skelet(anns[z])[:, 0]

            for i in range(len(CONNECT_INDEXES)):
                single = CONNECT_INDEXES[i]
                p_1 = all_kp[single[0]]
                p_2 = all_kp[single[1]]
                if p_1[0] != 0.0 and p_1[1] != 0.0 and p_2[0] != 0.0 and p_2[1] != 0:
                    plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color='r')
                    plt.scatter([p_1[0], p_2[0]], [p_1[1], p_2[1]], color='b')
        
        I = io.imread(os.path.join(self._image_folder_path, img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        plt.show()
    
    def save_records(self, prefix, images_pr, criteria=None, stop_after_part=None):
        """
        Parameters
        ----------
        prefix : str
            String containing path to save + prefix.
        images_pr : int
            Images per record. It is better to choose such a number that
            the records take about 200-300 mb of memory for the best pipeline
            performance.
        criteria : python function
            A function which takes keypoints masks and return a boolean. If True, the image
            is included. If `criteria` is set to None, a default criteria is taken.
        stop_after_part : int
            Generation of the tfrecords will be stopped after certain part,
            This parameters is usually used for creation a smaller tfrecord dataset
            (for example, for testing)
        """

        if criteria is None:
            criteria = self.__default_criteria

        ids_img = self._coco.getImgIds()
        count_images = len(ids_img)
        part = count_images // images_pr
        iterator = tqdm(range(count_images))

        image_tensors = []
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

            if len(anns) == 0 or len(anns) > self._max_people:
                continue

            all_kp = self.__take_default_skelet(anns[0])

            for people_n in range(1, len(anns)):
                all_kp_single = self.__take_default_skelet(anns[people_n])
                # all_kp - (n_keypoints, n_people, 3), concatenate by n_people axis
                all_kp = np.concatenate([all_kp, all_kp_single], axis=1)

            # Fill dimension n_people to maximum value according to self._max_people 
            # By placing zeros in other additional dimensions
            not_enougth = self._max_people - len(anns)
            zeros_arr = np.zeros([all_kp.shape[0], not_enougth, all_kp.shape[-1]]).astype(np.float32)
            all_kp = np.concatenate([all_kp, zeros_arr], axis=1)

            # Skip image if it's not suitable by critetia function
            if criteria is not None and not criteria(all_kp[..., -1:]):
                continue
            if len(image.shape) != 3:
                # Assume that is gray-scale image, so convert it to rgb
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image, all_kp = self.__rescale_image(image, all_kp)

            keypoints_tensors.append(all_kp[..., :2].astype(np.float32))
            keypoints_mask_tensors.append(all_kp[..., -1:].astype(np.float32))
            
            image_tensors.append(image.astype(np.float32))
            image_properties_tensors.append(np.array(image.shape).astype(np.float32))
            
            counter += 1
            
            # For large datasets, it's better to save them by parties
            if counter == part:
                print('Save part of the tfrecords...')
                record_mp_pose_estimation_train_data(
                    image_tensors=image_tensors,
                    keypoints_tensors=keypoints_tensors,
                    keypoints_mask_tensors=keypoints_mask_tensors,
                    image_properties_tensors=image_properties_tensors,
                    prefix=prefix + f'_{counter_saved}',
                    dp_per_record=images_pr
                )

                image_tensors = []
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
                keypoints_tensors=keypoints_tensors,
                keypoints_mask_tensors=keypoints_mask_tensors,
                image_properties_tensors=image_properties_tensors,
                prefix=prefix + f'_{counter_saved}',
                dp_per_record=images_pr
            )

        iterator.close()

    def __rescale_image(self, image, keypoints):
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
            # Ignore dimension of visibility of the keypoints 
            keypoints[..., :-1] *= scale
        
        return image, keypoints

    def __take_default_skelet(self, single_human_anns):
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
        else:
            foot_one = np.zeros(3).astype(np.float32)
        # Two
        if single_anns_foot[3][-1] > CocoPreparator.EPSILON and single_anns_foot[4][-1] > CocoPreparator.EPSILON:
            foot_two = (single_anns_foot[3] + single_anns_foot[4]) / 2.0
        else:
            foot_two = np.zeros(3).astype(np.float32)

        # Check neck
        if single_anns[5][-1] > CocoPreparator.EPSILON and single_anns[6][-1] > CocoPreparator.EPSILON:
            chest_p = (single_anns[5] + single_anns[6]) / 2.0

            face_p = np.zeros(3).astype(np.float32)
            div = 0
            for i in range(len(single_anns[:5])):
                if single_anns[i][-1] > CocoPreparator.EPSILON and single_anns[i][-1] > CocoPreparator.EPSILON:
                    div += 1
                    face_p += single_anns[i]

            face_p = face_p / div

            neck = (face_p + chest_p) / 2.0
            # Set visibility to 1.0 (i. e. visible)
            neck[-1] = 1.0
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

            avg_body_p_v1 = avg_body_p / div
            avg_body_p_v1[-1] = 1.0
        else:
            # Otherwise the point is not visible
            avg_body_p_v1 = np.zeros(3).astype(np.float32)

        all_kp_single = np.stack([
            avg_body_p_v1,
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
        # all_kp - (24, 3) ---> (24, 1, 3)
        all_kp_single = np.expand_dims(all_kp_single, axis=1)

        return all_kp_single


    def __default_criteria(self, keypoints_masks):
        """
        This criteria checking number of keypoints on the image and their relation to the number of all keypoints,
        All images with a lower relations will be thrown

        """
        # keypoints_masks - (n_keypoints, n_people, 1)
        check_ans = False
        n_keypoints = keypoints_masks.shape[0]
        keypoints_masks = np.transpose(keypoints_masks, axes=[1, 0, 2])
        # keypoints_masks - (n_people, n_keypoints, 1)
        check = np.max(np.sum(keypoints_masks > CocoPreparator.EPSILON, axis=1) / n_keypoints)

        if check > self.__criteria_throw:
            check_ans = True
        
        return check_ans

