from pose_estimation.model.pe_model import PEModel

from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import cv2
import json
import os

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


def create_prediction_coco_json(
        W: int,
        H: int,
        model: PEModel,
        ann_file_path: str,
        path_to_save: str,
        path_to_images: str,
        return_number_of_predictions=False):
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

    Return
    ------
    int
        Number of predictions, if `return_number_of_predictions` equal True
    """
    cocoGt = COCO(ann_file_path)
    cocoDt_json = []

    img_ids = cocoGt.getImgIds()

    iterator = tqdm(range(len(img_ids)))
    # Counter for generation unique IDs into annotation file
    counter = 0
    # Store batched images and image ids
    norm_img_list = []
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

        # Load image
        if path_to_images is None:
            readed_img = cv2.cvtColor(io.imread(single_img[COCO_URL]), cv2.COLOR_RGB2BGR)
        else:
            readed_img = cv2.imread(os.path.join(path_to_images, single_img[FILE_NAME]))
        source_img = cv2.resize(readed_img, (W, H))
        norm_img_list += [((source_img - 127.5) / 127.5).astype(np.float32)]
        image_ids_list += [single_ids]

        # Process batch of the images
        if batch_size == len(norm_img_list):
            humans_dict_list = model.predict(norm_img_list, resize_to=[W, H])

            for (single_humans_dict, single_image_ids) in zip(humans_dict_list, image_ids_list):
                for single_name in single_humans_dict:
                    single_elem = single_humans_dict[single_name]
                    cocoDt_json.append(
                        write_to_dict(
                            single_image_ids,
                            single_elem.score,
                            single_elem.to_list(),
                            counter
                        )
                    )

                    counter += 1
            # Clear batched arrays
            norm_img_list = []
            image_ids_list = []
    iterator.close()

    # Process images which remained
    uniq_images = len(norm_img_list)

    if uniq_images > 0:
        remain_images = batch_size - len(norm_img_list)
        norm_img_list += [norm_img_list[-1]] * remain_images

        humans_dict_list = model.predict(norm_img_list, resize_to=[W, H])[:uniq_images]

        for (single_humans_dict, single_image_ids) in zip(humans_dict_list, image_ids_list):
            for single_name in single_humans_dict:
                single_elem = single_humans_dict[single_name]
                cocoDt_json.append(
                    write_to_dict(
                        single_image_ids,
                        single_elem.score,
                        single_elem.to_list(),
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
