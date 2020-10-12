from pose_estimation.model.pe_model import PEModel

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
CATEGORY_ID = 'category_id'
SCORE = 'score'
COCO_URL = 'coco_url'
FILE_NAME = 'file_name'
DEFAULT_CATEGORY_ID = 1

DEFAULT_NUM_THREADES = 4

TYPE_THREAD = 'thread'
TYPE_PROCESS = 'process'


# Methods to process image with multiprocessing
def process_image(data):
    W, H, image_paths = data
    image = cv2.imread(image_paths)
    image = cv2.resize(image, (H, W))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image -= np.float32(127.5)
    image /= np.float32(127.5)
    return image


def start_process(image_paths: list, W: int, H: int, n_threade: int, type_parall: str):
    if type_parall == TYPE_THREAD:
        pool = dummy.Pool(processes=n_threade)
    elif type_parall == TYPE_PROCESS:
        pool = mp.Pool(processes=n_threade)
    else:
        raise TypeError(f'type {type_parall} is non known type for processing image in prediction writer!')

    res = pool.map(process_image, [(W, H, image_paths[index]) for index in range(len(image_paths))])

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
        type_parall=TYPE_THREAD
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
    # Counter for generation unique IDs into annotation file
    counter = 0
    # Store batched images and image ids
    imgs_path_list = []
    image_ids_list = []
    batch_size = model.get_batch_size()

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
            if type_parall is not None:
                norm_img_list = start_process(
                    imgs_path_list,
                    W=W, H=H,
                    n_threade=n_threade,
                    type_parall=type_parall
                )
            else:
                norm_img_list = [
                    process_image((W, H, imgs_path_list[index]))
                    for index in range(len(imgs_path_list))
                ]

            humans_predicted_list = model.predict(norm_img_list, resize_to=[H, W])

            for (single_humans_predicted_list, single_image_ids) in zip(humans_predicted_list, image_ids_list):
                for single_prediction in single_humans_predicted_list:
                    cocoDt_json.append(
                        write_to_dict(
                            single_image_ids,
                            single_prediction.score,
                            single_prediction.to_list(),
                            counter
                        )
                    )

                    counter += 1
            # Clear batched arrays
            imgs_path_list = []
            image_ids_list = []
    iterator.close()

    # Process images which remained
    uniq_images = len(imgs_path_list)

    if uniq_images > 0:
        remain_images = batch_size - uniq_images
        imgs_path_list += [imgs_path_list[-1]] * remain_images

        norm_img_list = start_process(
            imgs_path_list,
            W=W, H=H,
            n_threade=n_threade,
            type_parall=type_parall
        )

        humans_predicted_list = model.predict(norm_img_list, resize_to=[W, H])[:uniq_images]

        for (single_humans_predicted_list, single_image_ids) in zip(humans_predicted_list, image_ids_list):
            for single_prediction in single_humans_predicted_list:
                cocoDt_json.append(
                    write_to_dict(
                        single_image_ids,
                        single_prediction.score,
                        single_prediction.to_list(),
                        counter
                    )
                )

                counter += 1

    with open(path_to_save, 'w') as fp:
        json.dump(cocoDt_json, fp)

    if return_number_of_predictions:
        return len(cocoDt_json)


def write_to_dict(img_id: int, score: float, maki_keypoints: list, id: int) -> dict:
    """
    Write data into dict for saving it later into JSON

    """
    return {
        CATEGORY_ID: DEFAULT_CATEGORY_ID,
        IMAGE_ID: img_id,
        SCORE: score,
        KEYPOINTS: maki_keypoints,
        ID: id,
    }
