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

from pose_estimation.model.pe_model import PEModel
from pose_estimation.utils.nns_tools.preprocess import preprocess_input, TF
from pose_estimation.utils.nns_tools.keypoint_tools import scale_predicted_kp
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import cv2
import json
import os
import traceback


# Write prediction from models into json file (in coco annotations style)

"""
Example of the json how to save single prediction

{
    "category_id": 1,
    "image_id": int,
    "score": float,
    "keypoints": list of shape (24, 3), last dimension fill with 1
}

"""

from pose_estimation.data_preparation.coco_wholebody_relayout import IMAGE_ID, KEYPOINTS, ID
from pose_estimation.utils import scales_image_single_dim_keep_dims

CATEGORY_ID = 'category_id'
SCORE = 'score'
COCO_URL = 'coco_url'
FILE_NAME = 'file_name'
DEFAULT_CATEGORY_ID = 1

PADDING = 8


# Methods to process image with multiprocessing
def process_image(
        min_size_h: int,
        image_paths: str,
        mode: str,
        div: float,
        shift: float,
        use_bgr2rgb: bool) -> tuple:
    image = cv2.imread(image_paths)
    source_size = image.shape[:-1]

    # First scale image like in preparation of training data
    # We keep H with certain size and scale other (keesp relation)
    x_scale, y_scale = scales_image_single_dim_keep_dims(
        image_size=image.shape[:-1],
        resize_to=min_size_h
    )

    new_H, new_W = (round(y_scale * image.shape[0]), round(x_scale * image.shape[1]))

    image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
    input_image_size_into_model = (new_H, new_W)

    # To keep human view better, padding with zeros
    # In order to new_W be divided by 8 without remainder
    if new_W % PADDING != 0:
        padding_image = np.zeros((new_H, new_W + (PADDING - new_W % PADDING), 3)).astype(np.float32, copy=False)
        padding_image[:, :new_W] = image
        image = padding_image

    if use_bgr2rgb:
        image = image[..., ::-1]

    if mode is not None:
        image = preprocess_input(image, mode=mode)
    elif div is not None and shift is not None:
        image /= div
        image -= shift

    return source_size, input_image_size_into_model, image.astype(np.float32, copy=False)


def create_prediction_coco_json(
        min_size_h: int,
        model: PEModel,
        ann_file_path: str,
        path_to_save: str,
        path_to_images: str,
        return_number_of_predictions=False,
        mode=TF,
        divider=None,
        shift=None,
        use_bgr2rgb=False,
    ):
    """
    Create prediction JSON for evaluation on COCO dataset

    Parameters
    ---------
    min_size_h : int
        Min size of Height, which was used in preparation of data
    model : pe_model
        Model from which collects prediction.
        Model should be built and initialized with session.
    ann_file_path : str
        Path to annotation file on which will be used evaluation.
        It can be original COCO file or re-layout file.
    path_to_save : str
        Path to save prediction JSON.
        Example: /user/exp/predicted_data.json
    path_to_images : str
        Folder to image that will be loaded to estimate poses.
        Should be fit to annotation file (i. e. validation JSON to validation images)
    return_number_of_predictions : bool
        If equal True, will be returned number of prediction
    mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
        - tf: will scale pixels between -1 and 1,
          sample-wise.
        - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.
        But for control normalization, set this value to None and use `divider` and `shift`
    divider : float
        Value to divide image,
        By default equal to None, i.e. will be not used.
        NOTICE! `mode` parameter has bigger priority than this value,
                i.e. while mode does not equal to None, this value never be used.
    shift : float
        Value to shift image (minus is taken into account),
        By default equal to None, i.e. will be not used.
        NOTICE! `mode` parameter has bigger priority than this value,
                i.e. while mode does not equal to None, this value never be used.
    use_bgr2rgb : bool
        If equal to true, loaded images will be converted into rgb,
        otherwise they will be in bgr format

    Return
    ------
    int
        Number of predictions, if `return_number_of_predictions` equal True
    """

    # Methods to process image with multiprocessing

    cocoGt = COCO(ann_file_path)
    cocoDt_json = []

    img_ids = cocoGt.getImgIds()

    iterator = tqdm(range(len(img_ids)))
    # Store batched images and image ids
    imgs_path_list = []
    image_ids_list = []
    batch_size = model.get_batch_size()
    try:
        for i in iterator:
            single_ids = img_ids[i]
            # Take single image
            single_img = cocoGt.loadImgs(single_ids)[0]
            annIds = cocoGt.getAnnIds(imgIds=single_img[ID], iscrowd=None)
            anns = cocoGt.loadAnns(annIds)
            # Ignore images where are no people
            if len(anns) == 0:
                continue

            # Store path to img and its ids
            imgs_path_list.append(os.path.join(path_to_images, single_img[FILE_NAME]))
            image_ids_list += [single_ids]

            # Process batch of the images
            if batch_size == len(imgs_path_list):
                get_batched_result(
                    cocoDt_json=cocoDt_json,
                    image_ids_list=image_ids_list,
                    imgs_path_list=imgs_path_list,
                    min_size_h=min_size_h,
                    model=model,
                    mode=mode,
                    divider=divider,
                    shift=shift,
                    use_bgr2rgb=use_bgr2rgb
                )
                # Clear batched arrays
                imgs_path_list = []
                image_ids_list = []
        iterator.close()

        # Process images which remained
        uniq_images = len(imgs_path_list)

        if uniq_images > 0:
            remain_images = batch_size - uniq_images
            imgs_path_list += [imgs_path_list[-1]] * remain_images

            get_batched_result(
                cocoDt_json=cocoDt_json,
                image_ids_list=image_ids_list,
                imgs_path_list=imgs_path_list,
                min_size_h=min_size_h,
                model=model,
                mode=mode,
                divider=divider,
                shift=shift,
                use_bgr2rgb=use_bgr2rgb,
            )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
    finally:
        with open(path_to_save, 'w') as fp:
            json.dump(cocoDt_json, fp)

        if return_number_of_predictions:
            return len(cocoDt_json)


def get_batched_result(
        cocoDt_json: list,
        image_ids_list: list,
        imgs_path_list: list,
        min_size_h: int,
        model: PEModel,
        mode=TF,
        divider=None,
        shift=None,
        use_bgr2rgb=False):
    """
    Load certain images, get output from model and write it into dict with certain format

    """

    source_size_and_model_size_and_norm_img_list = [
        process_image(
            min_size_h=min_size_h,
            image_paths=imgs_path_list[index],
            mode=mode,
            shift=shift,
            div=divider,
            use_bgr2rgb=use_bgr2rgb
        )
        for index in range(len(imgs_path_list))
    ]

    norm_image_list = []
    source_size_list = []
    model_size_input_list = []
    for i in range(len(source_size_and_model_size_and_norm_img_list)):
        # Separate image and their source size (prediction will be scaled to source size)
        source_size_list.append(source_size_and_model_size_and_norm_img_list[i][0])        # source size
        model_size_input_list.append(source_size_and_model_size_and_norm_img_list[i][1])   # model size
        norm_image_list.append(source_size_and_model_size_and_norm_img_list[i][2])         # image

    humans_predicted_list = [
        model.predict(single_norm_image_list)
        for single_norm_image_list in norm_image_list
    ]

    for (source_size_single, model_size_input_single, single_humans_predicted_list, single_image_ids) in zip(
            source_size_list,
            model_size_input_list,
            humans_predicted_list,
            image_ids_list
    ):
        # Scale prediction for single image
        scale_predicted_kp(
            predictions=[single_humans_predicted_list],
            model_size=model_size_input_single,
            source_size=source_size_single
        )
        for single_prediction in single_humans_predicted_list:
            cocoDt_json.append(
                write_to_dict(
                    single_image_ids,
                    single_prediction.score,
                    single_prediction.to_list()
                )
            )


def write_to_dict(img_id: int, score: float, maki_keypoints: list) -> dict:
    """
    Write data into dict for saving it later into JSON

    """
    return {
        CATEGORY_ID: DEFAULT_CATEGORY_ID,
        IMAGE_ID: img_id,
        SCORE: score,
        KEYPOINTS: maki_keypoints
    }
