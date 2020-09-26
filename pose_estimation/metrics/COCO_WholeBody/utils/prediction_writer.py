from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import cv2
import json

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

from .relayout_coco_annotation import IMAGE_ID, MAKI_KEYPOINTS
CATEGORY_ID = 'category_id'
SCORE = 'score'
COCO_URL = 'coco_url'
DEFAULT_CATEGORY_ID = 1


def create_prediction_coco_json(W: int, H: int, model, ann_file_path: str, path_to_save: str):
    cocoGt = COCO(ann_file_path)
    cocoDt_json = []

    iterator = tqdm(range(len(cocoGt.getImgIds())))

    for i in iterator:
        single_ids = cocoGt.getImgIds()[i]
        # Take single image
        single_img = cocoGt.loadImgs(single_ids)[0]

        # Load image
        source_img = cv2.cvtColor(io.imread(single_img[COCO_URL], cv2.COLOR_RGB2BGR))
        source_img = cv2.resize(source_img, (W, H))
        norm_img = [((source_img - 127.5) / 127.5).astype(np.float32)]
        # Predict and take only single
        humans_dict = model.predict(norm_img)[0]

        for single_name in humans_dict:
            single_elem = humans_dict[single_name]
            cocoDt_json.append(
                write_to_dict(single_ids, single_elem.score, single_elem.to_list())
            )

    iterator.close()

    with open(path_to_save, 'w') as fp:
        json.dump(cocoDt_json, fp)


def write_to_dict(img_id: int, score: float, maki_keypoints: list) -> dict:
    return {
        CATEGORY_ID: DEFAULT_CATEGORY_ID,
        IMAGE_ID: img_id,
        SCORE: score,
        MAKI_KEYPOINTS: maki_keypoints,
    }

