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
from pose_estimation.utils.preprocess import preprocess_input, TF
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import cv2
import json
import os
from multiprocessing import dummy
import multiprocessing as mp

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

from .relayout_coco_annotation import IMAGE_ID, KEYPOINTS, ID
from .utils import rescale_image

CATEGORY_ID = 'category_id'
SCORE = 'score'
COCO_URL = 'coco_url'
FILE_NAME = 'file_name'
DEFAULT_CATEGORY_ID = 1

TYPE_THREAD = 'thread'
TYPE_PROCESS = 'process'


# Methods to process image with multiprocessing
def process_image(data):
    W, H, image_paths, mode, div, shift, use_bgr2rgb, use_force_resize = data
    image = cv2.imread(image_paths)
    x_scale, y_scale = rescale_image(
        image_size=[image.shape[0], image.shape[1]],
        min_image_size=[H, W],
        resize_to=[None, None],
        use_force_resize=use_force_resize
    )
    new_H, new_W = (int(y_scale * image.shape[0]), int(x_scale * image.shape[1]))
    image = cv2.resize(image, (new_W, new_H))

    if use_bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mode is not None:
        image = preprocess_input(image, mode=mode)
    elif div is not None and shift is not None:
        image /= div
        image -= shift

    return image.astype(np.float32, copy=False)


def start_process(
        image_paths: list,
        W: int, H: int,
        n_threade: int,
        type_parall: str,
        mode: str,
        divider : float,
        shift: float,
        use_bgr2rgb: bool,
        use_force_resize: bool
    ):
    if type_parall == TYPE_THREAD:
        pool = dummy.Pool(processes=n_threade)
    elif type_parall == TYPE_PROCESS:
        pool = mp.Pool(processes=n_threade)
    else:
        raise TypeError(f'type {type_parall} is non known type for processing image in prediction writer!')

    res = pool.map(
        process_image,
        [
            (W, H, image_paths[index], mode, divider, shift, use_bgr2rgb, use_force_resize)
            for index in range(len(image_paths))
        ]
    )

    pool.close()
    pool.join()

    return res


def create_prediction_coco_json(
        W: int,
        H: int,
        model: PEModel,
        ann_file_path: str,
        path_to_save: str,
        path_to_images: str,
        return_number_of_predictions=False,
        n_threade=None,
        type_parall=TYPE_THREAD,
        mode=TF,
        divider=None,
        shift=None,
        use_bgr2rgb=False,
        use_force_resize=True
    ):
    """
    Create prediction JSON for evaluation on COCO dataset

    Parameters
    ----------
    W : int
        Width of the image
    H : int
        Height of the image
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
    n_threade : int
        Number of threades to process image (resize, normalize and etc...),
        By default parallel calculation not used, i.e. value equal to None
    type_parall : str
        Type of the parallel calculation for loading and preprocessing images,
        Can be `thread` or `process` values.
        By default equal to `thread`
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
                    W=W, H=H,
                    model=model,
                    n_threade=n_threade,
                    type_parall=type_parall,
                    mode=mode,
                    divider=divider,
                    shift=shift,
                    use_bgr2rgb=use_bgr2rgb,
                    use_force_resize=use_force_resize
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
                W=W, H=H,
                model=model,
                n_threade=n_threade,
                type_parall=type_parall,
                mode=mode,
                divider=divider,
                shift=shift,
                use_bgr2rgb=use_bgr2rgb,
                use_force_resize=use_force_resize
            )
    except Exception as ex:
        print(ex)
    finally:
        with open(path_to_save, 'w') as fp:
            json.dump(cocoDt_json, fp)

        if return_number_of_predictions:
            return len(cocoDt_json)


def get_batched_result(
        cocoDt_json: list,
        image_ids_list: list,
        imgs_path_list: list,
        W: int, H: int,
        model: PEModel,
        n_threade=None,
        type_parall=TYPE_THREAD,
        mode=TF,
        divider=None,
        shift=None,
        use_bgr2rgb=False,
        use_force_resize=True):
    """
    Load certain images, get output from model and write it into dict with certain format

    """
    if type_parall is not None:
        norm_img_list = start_process(
            imgs_path_list,
            W=W, H=H,
            n_threade=n_threade,
            type_parall=type_parall,
            mode=mode,
            shift=shift,
            divider=divider,
            use_bgr2rgb=use_bgr2rgb,
            use_force_resize=use_force_resize
        )
    else:
        norm_img_list = [
            process_image(
                (
                    W, H,
                    imgs_path_list[index],
                    mode,
                    shift,
                    divider,
                    use_bgr2rgb,
                    use_force_resize
                )
            )
            for index in range(len(imgs_path_list))
        ]

    humans_predicted_list = model.predict(norm_img_list)

    for (single_humans_predicted_list, single_image_ids) in zip(humans_predicted_list, image_ids_list):
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
