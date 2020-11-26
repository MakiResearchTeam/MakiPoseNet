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

from pose_estimation.trainers.tester import Tester
from pose_estimation.utils.nns_tools.keypoint_tools import scale_predicted_kp
from pose_estimation.metrics.COCO_WholeBody import (MYeval_wholebody,
                                                    create_prediction_coco_json)
from pose_estimation.utils.visual_tools.constants import CONNECT_INDEXES
from pose_estimation.utils import SkeletonDrawer, preprocess_input, visualize_paf, draw_skeleton
from makiflow.tools.video.video_reader import VideoReader

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class CocoTester(Tester):
    TEST_IMAGE = 'test_image'
    TRAIN_N = 'train_{}'
    TEST_N = "test_{}"

    _EXCEPTION_IMAGE_WAS_NOT_FOUND = "Image by path {0} was not found!"

    HEATMAP_CENTER_BODY_IMAGE = 'heatmap_center_body_image'
    HEATMAP_CENTER_BODY_TEXT = 'center body'
    HEATMAP_CENTER_BODY_INDEX = 0

    HEATMAP_LEFT_SHOULDER_IMAGE = 'heatmap_left_shoulder_image'
    HEATMAP_LEFT_SHOULDER_TEXT = "left shoulder"
    HEATMAP_LEFT_SHOULDER_INDEX = 6

    HEATMAP_RIGHT_SHOULDER_IMAGE = 'heatmap_right_shoulder_image'
    HEATMAP_RIGHT_SHOULDER_TEXT = "right shoulder"
    HEATMAP_RIGHT_SHOULDER_INDEX = 7

    HEATMAP_RIGHT_HAND_IMAGE = 'heatmap_right_hand_image'
    HEATMAP_RIGHT_HAND_TEXT = "right hand"
    HEATMAP_RIGHT_HAND_INDEX = 11

    HEATMAP_LEFT_HAND_IMAGE = 'heatmap_left_hand_image'
    HEATMAP_LEFT_HAND_TEXT = "left hand"
    HEATMAP_LEFT_HAND_INDEX = 10

    DRAW_LIST = [
        [HEATMAP_CENTER_BODY_TEXT, HEATMAP_CENTER_BODY_INDEX],
        [HEATMAP_LEFT_SHOULDER_TEXT, HEATMAP_LEFT_SHOULDER_INDEX],
        [HEATMAP_RIGHT_SHOULDER_TEXT, HEATMAP_RIGHT_SHOULDER_INDEX],
        [HEATMAP_RIGHT_HAND_TEXT, HEATMAP_RIGHT_HAND_INDEX],
        [HEATMAP_LEFT_HAND_TEXT, HEATMAP_LEFT_HAND_INDEX]
    ]

    PAFF_IMAGE = 'paff_image'
    SKELETON = 'skeleton'
    ITERATION_COUNTER = 'iteration_counter'

    AP_IOU_050 = "AP with IOU 0.50"
    AR_IOU_050 = "AR with IOU 0.50"

    _CENTRAL_SIZE = 600
    _LENGHT_VIDEO = 600
    _ZERO_VALUE = 0.0

    def _init(self):
        # Test images
        self.__init_test_images()
        # Train images
        self.__init_train_images()
        self.add_scalar(CocoTester.ITERATION_COUNTER)

        if self.cocoGt is not None:
            self.add_scalar(CocoTester.AP_IOU_050)
            self.add_scalar(CocoTester.AR_IOU_050)

    def __init_train_images(self):

        self._norm_images_train = []
        self._names_train = []

        for i in range(len(self._train_images)):
            single_train_image = self._train_images[i]
            if not self._use_bgr2rgb:
                single_train_image = cv2.cvtColor(single_train_image, cv2.COLOR_BGR2RGB)

            single_train_image = cv2.resize(single_train_image, (self.W, self.H))
            self._norm_images_train.append(
                self.__preprocess(single_train_image)
            )

            self._names_train.append(CocoTester.TRAIN_N.format(i))
            # Image with skeletons
            self.add_image(self._names_train[-1], n_images=1)

    def __init_test_images(self):
        if not isinstance(self._config[CocoTester.TEST_IMAGE], list):
            test_images_path = [self._config[CocoTester.TEST_IMAGE]]
        else:
            test_images_path = self._config[CocoTester.TEST_IMAGE]

        self._norm_images = []
        self._test_images = []
        self._names = []

        for i, single_path in enumerate(test_images_path):
            test_image = cv2.imread(single_path)
            if test_image is None:
                raise TypeError(CocoTester._EXCEPTION_IMAGE_WAS_NOT_FOUND.format(self._config[CocoTester.TEST_IMAGE]))

            test_image = cv2.resize(test_image, (self.W, self.H))
            if self._use_bgr2rgb:
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            self._norm_images.append(
                self.__preprocess(test_image)
            )

            # The image has to have batch dimension
            self._test_images.append(test_image.astype(np.uint8))
            self._names.append(CocoTester.TEST_N.format(i))
            # Plus: original image, image with paf, image with skeleton = 3
            self.add_image(self._names[-1], n_images=len(CocoTester.DRAW_LIST) + 3)

    def evaluate(self, model, iteration):

        dict_summary_to_tb = {CocoTester.ITERATION_COUNTER: iteration}
        # Skip skeleton drawing if network accuracy is low
        is_network_good_right_now = False
        new_log_folder = None

        # Metric stuff
        if self.cocoGt is not None:
            # Create folder to store AP/AR results
            new_log_folder = os.path.join(self._path_to_save_logs, f"iter_{iteration}")
            os.makedirs(new_log_folder, exist_ok=True)
            save_predicted_json = os.path.join(new_log_folder, self.NAME_PREDICTED_ANNOT_JSON)

            num_detections = create_prediction_coco_json(
                self.W, self.H, model,
                self._path_to_relayout_annot,
                save_predicted_json,
                self._path_to_val_images,
                return_number_of_predictions=True,
                n_threade=self._n_threade,
                type_parall=self._type_parall,
                mode=self._norm_mode,
                divider=self._norm_div,
                shift=self._norm_shift,
                use_bgr2rgb=self._use_bgr2rgb,
                use_force_resize=False
            )
            # Process evaluation only if number of detection bigger that 0
            if num_detections > 0:
                cocoDt = self.cocoGt.loadRes(save_predicted_json)
                cocoEval = MYeval_wholebody(cocoDt=cocoDt, cocoGt=self.cocoGt)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                with open(os.path.join(new_log_folder, self.AP_AR_DATA_TXT), 'w') as fp:
                    fp.write(cocoEval.get_stats_str())

                # Take and write AP/AR values
                a_prediction, a_recall = cocoEval.get_AP_AR_with_IoU_50()
                dict_summary_to_tb.update({CocoTester.AP_IOU_050: a_prediction})
                dict_summary_to_tb.update({CocoTester.AR_IOU_050: a_recall})

                # Set flag to True, if AP bigger than 1 percent, i.e. we can create test video
                if a_prediction > 0.01:
                    is_network_good_right_now = True

            else:
                # If there is no detection, write zero values
                dict_summary_to_tb.update({CocoTester.AP_IOU_050: self._ZERO_VALUE})
                dict_summary_to_tb.update({CocoTester.AR_IOU_050: self._ZERO_VALUE})

        # Draw test images
        # Write heatmap,paf and image itself for each image in `_test_images`
        self.__get_test_tb_data(model, dict_summary_to_tb, is_network_good_right_now, new_log_folder)
        # Draw train images
        self.__get_train_tb_data(model, dict_summary_to_tb, is_network_good_right_now)
        # Write data into tensorBoard
        self.write_summaries(
            summaries=dict_summary_to_tb,
            step=iteration
        )

    def draw_heatmap(self, heatmap, name_heatmap, shift_image=60, dpi=80):
        h, w = heatmap.shape

        figsize = w / float(dpi), h / float(dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        sns.heatmap(heatmap)
        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = np.reshape(data, (h, w, 3))

        plt.close('all')

        return self.__put_text_on_image(data, name_heatmap, shift_image)

    def __get_train_tb_data(self, model, dict_summary_to_tb, is_network_good_right_now=True):
        for i, (single_norm_train, single_train) in enumerate(zip(self._norm_images_train, self._train_images)):
            # Copy image to keep it clear (without painted lines)
            drawed_image = single_train.copy()
            # Draw skeletons
            if is_network_good_right_now:
                # Draw prediction
                # Take single prediction for one image and take it
                prediction = model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0]

                # Feed list of predictions
                # Scale predicted keypoint into original image size and paint them
                scale_predicted_kp([prediction], (self.H, self.W), single_train.shape[:2])
                drawed_image = draw_skeleton(drawed_image, prediction, CONNECT_INDEXES, color=(255, 0, 0))

            # Draw ground-truth
            drawed_image = draw_skeleton(drawed_image, self._ground_truth[i], CONNECT_INDEXES, color=(40, 255, 40))

            dict_summary_to_tb.update({self._names_train[i]: np.expand_dims(drawed_image, axis=0).astype(np.uint8)})

    def __get_test_tb_data(self, model, dict_summary_to_tb, is_network_good_right_now, new_log_folder=None):
        for i, (single_norm, single_test) in enumerate(zip(self._norm_images, self._test_images)):
            single_batch = [self.__put_text_on_image(single_test, self._names[i])]
            peaks, heatmap, paf = model.predict(
                np.stack([single_norm] * model.get_batch_size(), axis=0),
                using_estimate_alg=False,
                resize_to=[self.H, self.W]
            )

            drawed_paff = self.__put_text_on_image(
                    visualize_paf(single_test, paf[0]),
                    self._names[i] + '_' + CocoTester.PAFF_IMAGE
            )
            single_batch.append(drawed_paff)

            for single_draw_params in CocoTester.DRAW_LIST:
                index = single_draw_params[1]
                name_p = single_draw_params[0]

                single_batch.append(
                    self.draw_heatmap(
                        heatmap[0][..., index],
                        name_p
                    )
                )

            # Draw skeletons
            if is_network_good_right_now:
                predictions = model.predict(
                    np.stack([single_norm] * model.get_batch_size(), axis=0),
                    resize_to=[self.H, self.W]
                )[0]
                drawed_image = draw_skeleton(single_test.copy(), predictions, CONNECT_INDEXES)
                single_batch.append(self.__put_text_on_image(drawed_image, self.SKELETON))

            dict_summary_to_tb.update({self._names[i]: np.stack(single_batch, axis=0).astype(np.uint8)})

        # Create video with skeletons
        # If AP were bigger than 1%
        if is_network_good_right_now:
            self.__record_prediction_video(model, new_log_folder)

    def __record_prediction_video(self, model, new_log_folder):
        # Read/Gives us frames from certain video
        video_r = VideoReader(self._video_path)

        # Function for preprocessing images before put them into model
        def transform(batch_images, m_w, m_h, mode, use_bgr2rgb, func_preprocess=None):
            new_images = []
            for i in range(len(batch_images)):
                single_image = batch_images[i]
                single_image = cv2.resize(single_image, (m_w, m_h))
                if use_bgr2rgb:
                    single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)

                if mode is not None:
                    single_image = preprocess_input(single_image, mode=mode)
                else:
                    single_image = func_preprocess(single_image)

                new_images.append(single_image)
            return new_images

        # By default model has batch size 1 in inference mode
        video_generator = video_r.get_iterator(1)

        # Save video in logs folder if save folder were not provided in config
        if self._save_pred_video_folder is None:
            save_folder = os.path.join(new_log_folder, self.VIDEO_TEST.format(self._video_counter))
        else:
            save_folder = os.path.join(self._save_pred_video_folder, self.VIDEO_TEST.format(self._video_counter))

        # Update counter, we use it to give new video unique name
        # For example if videos are saved in one folder
        self._video_counter += 1

        # Create obj to draw skeletons
        video_drawer = SkeletonDrawer(save_folder)

        # Start predict and draw
        for i, batch_image in enumerate(video_generator):

            # After certain frames, break
            if i == self._LENGHT_VIDEO:
                break

            # Transform batch
            transformed_image_batch = transform(
                batch_image,
                m_h=self.H, m_w=self.W,
                mode=self._norm_mode,
                use_bgr2rgb=self._use_bgr2rgb,
                func_preprocess=self.__preprocess
            )
            predictions = model.predict(transformed_image_batch)

            # scale predictions into original size
            scale_predicted_kp(predictions, (self.H, self.W), batch_image[0].shape[:2])
            # draw predictions
            video_drawer.write(batch_image, predictions)

        # Close write stream
        video_drawer.release()

    def __put_text_on_image(self, image, text, shift_image=60):
        h,w = image.shape[:-1]
        img = np.ones((h + shift_image, w, 3)) * 255.0
        img[:h, :w] = image

        cv2.putText(
            img,
            text,
            (shift_image // 4, h + shift_image // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(h / self._CENTRAL_SIZE, w / self._CENTRAL_SIZE),
            (0, 0, 0),
            1
        )

        return img.astype(np.uint8)

    def __preprocess(self, image: np.ndarray):
        image = image.copy()

        if self._norm_mode is not None:
            norm_image = preprocess_input(
                    image,
                    mode=self._norm_mode
            ).astype(np.float32)

        elif self._norm_div is not None and self._norm_shift is not None:
            norm_image = (image / self._norm_div - self._norm_shift).astype(np.float32)
        else:
            norm_image = image.astype(np.float32)

        return norm_image

