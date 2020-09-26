import numpy as np
from tqdm import tqdm
import copy
import json
from pycocotools.coco import COCO

from pose_estimation.data_preparation.coco_preparator_api import CocoPreparator
from pose_estimation.metrics.COCO_WholeBody.eval import MAKI_KEYPOINTS


# Annotations in JSON
ANNOTATIONS = 'annotations'
IMAGES = 'images'
BBOX = 'bbox'
SEGMENTATION = 'segmentation'
AREA = 'area'
# Stored in the annotations
IMAGE_ID = 'image_id'
# Id in the image
ID = 'id'

# Images in JSON
HEIGHT = 'height'
WIDTH = 'width'


def relayout_keypoints(W: int, H: int, ann_file_path: str, path_to_save: str):
    with open(ann_file_path, 'r') as fp:
        cocoGt_json = json.load(fp)
    cocoGt = COCO(ann_file_path)

    Maki_cocoGt_json = copy.deepcopy(cocoGt_json)
    iterator = tqdm(range(len(cocoGt_json[ANNOTATIONS])))

    for i in iterator:
        single_anns = cocoGt_json[ANNOTATIONS][i]
        new_keypoints = CocoPreparator.take_default_skelet(single_anns)
        image_annot = find_image_annot(cocoGt_json, single_anns[IMAGE_ID])
        if image_annot is None:
            raise ModuleNotFoundError(f'Image id: {single_anns[IMAGE_ID]} was not found.')

        image_size = (image_annot[HEIGHT], image_annot[WIDTH])

        scale_k = (W / image_size[1], H / image_size[0], 1)
        scale_bbox = (W / image_size[1], H / image_size[0])

        new_keypoints = (new_keypoints.reshape(-1, 3) * scale_k).reshape(-1).astype(np.float32).tolist()
        new_bbox = (np.array(single_anns[BBOX]).reshape(2, 2) * scale_bbox).reshape(-1).tolist()

        new_segmentation = single_anns[SEGMENTATION]
        for i in range(len(new_segmentation)):
            single_new_seg_coord = np.array(new_segmentation[i]).reshape(-1, 2) * scale_bbox
            single_new_seg_coord = single_new_seg_coord.astype(np.float32).reshape(-1).tolist()
            new_segmentation[i] = single_new_seg_coord

        # Calculate new value for 'area'
        new_area = float(np.sum(
            cocoGt.annToMask({
                SEGMENTATION: new_segmentation,
                IMAGE_ID: single_anns[IMAGE_ID]
            })
        ))

        Maki_cocoGt_json[ANNOTATIONS][i][MAKI_KEYPOINTS] = new_keypoints
        Maki_cocoGt_json[ANNOTATIONS][i][BBOX] = new_bbox
        Maki_cocoGt_json[ANNOTATIONS][i][SEGMENTATION] = new_segmentation
        Maki_cocoGt_json[ANNOTATIONS][i][AREA] = new_area

    with open(path_to_save, 'w') as fp:
        json.dump(Maki_cocoGt_json, fp)


def find_image_annot(cocoGt_json: dict, img_id: int) -> dict:
    for single_annot in cocoGt_json[IMAGES]:
        if single_annot[ID] == img_id:
            return single_annot

    return None
