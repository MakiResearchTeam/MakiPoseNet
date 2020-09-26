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
    "maki_keypoints": list of shape (24, 3), last dimension fill with 1
}

"""

from .relayout_coco_annotation import IMAGE_ID, MAKI_KEYPOINTS, ANNOTATIONS, ID, AREA
CATEGORY_ID = 'category_id'
SCORE = 'score'
COCO_URL = 'coco_url'
FILE_NAME = 'file_name'
DEFAULT_CATEGORY_ID = 1

BIG_NUMBER = 10 ** 10
LOW_NUMBER = -1
EPSILONE = 1e-2


def create_prediction_coco_json(W: int, H: int, model, ann_file_path: str, path_to_save: str, path_to_images: str):
    cocoGt = COCO(ann_file_path)
    cocoDt_json = []

    img_ids = cocoGt.getImgIds()

    iterator = tqdm(range(len(img_ids)))
    counter = 0

    for i in iterator:
        single_ids = img_ids[i]
        # Take single image
        single_img = cocoGt.loadImgs(single_ids)[0]
        annIds = cocoGt.getAnnIds(imgIds=single_img['id'], iscrowd=None)
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
        norm_img = [((source_img - 127.5) / 127.5).astype(np.float32)]
        # Predict and take only single
        humans_dict = model.predict(norm_img)[0]

        for single_name in humans_dict:
            single_elem = humans_dict[single_name]
            cocoDt_json.append(
                write_to_dict(
                    single_ids,
                    single_elem.score,
                    single_elem.to_list(),
                    counter,
                    calculate_area(np.array(single_elem.to_list()).reshape(-1, 3))
                )
            )

            counter += 1

    iterator.close()

    with open(path_to_save, 'w') as fp:
        json.dump({ANNOTATIONS: cocoDt_json}, fp)


def write_to_dict(img_id: int, score: float, maki_keypoints: list, id: int, area: float) -> dict:
    return {
        CATEGORY_ID: DEFAULT_CATEGORY_ID,
        IMAGE_ID: img_id,
        SCORE: score,
        MAKI_KEYPOINTS: maki_keypoints,
        ID: id,
        AREA: area
    }


def calculate_area(maki_keypoints: np.ndarray):
    left_cor = [BIG_NUMBER, BIG_NUMBER]
    right_cor = [LOW_NUMBER, LOW_NUMBER]

    for i in range(len(maki_keypoints)):
        if maki_keypoints[i][0] > EPSILONE:
            left_cor[0] = min(left_cor[0], maki_keypoints[i][0])
            right_cor[0] = max(right_cor[0], maki_keypoints[i][0])

        if maki_keypoints[i][1] > EPSILONE:
            left_cor[1] = min(left_cor[1], maki_keypoints[i][1])
            right_cor[1] = max(right_cor[1], maki_keypoints[i][1])

    return (right_cor[0] - left_cor[0]) * (right_cor[1] - left_cor[1])