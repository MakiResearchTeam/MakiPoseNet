import numpy as np
from tqdm import tqdm
import copy
import json
from pycocotools.coco import COCO

from pose_estimation.data_preparation.coco_preparator_api import CocoPreparator
from .utils import rescale_image


# Annotations in JSON
ANNOTATIONS = 'annotations'
IMAGES = 'images'
BBOX = 'bbox'
SEGMENTATION = 'segmentation'
KEYPOINTS = 'keypoints'
AREA = 'area'
# Stored in the annotations
IMAGE_ID = 'image_id'
# Id in the image
ID = 'id'

# Images in JSON
HEIGHT = 'height'
WIDTH = 'width'


EPS = 1e-3


def relayout_keypoints(
        W: int, H: int,
        ann_file_path: str,
        path_to_save: str,
        limit_number=None,
        mode_area_calculation=SEGMENTATION,
        min_w_size=None,
        min_h_size=None,
        use_force_resize=True
    ):

    with open(ann_file_path, 'r') as fp:
        cocoGt_json = json.load(fp)
    cocoGt = COCO(ann_file_path)

    # Store: (image_id, image_info)
    dict_id_by_image_info = dict([(elem[ID], elem) for elem in cocoGt_json[IMAGES]])
    # Store: (image_id, bool)
    used_ids = dict([(elem[ID], False) for elem in cocoGt_json[IMAGES]])

    Maki_cocoGt_json = copy.deepcopy(cocoGt_json)
    # Clear information ablut annotations and images
    # In next for loop, we write new information
    Maki_cocoGt_json[ANNOTATIONS] = []
    Maki_cocoGt_json[IMAGES] = []

    if limit_number is None:
        iterator = tqdm(range(len(cocoGt_json[ANNOTATIONS])))

    elif type(limit_number) == int:
        iterator = tqdm(range(  min(limit_number, len(cocoGt_json[ANNOTATIONS])) ))

    else:
        raise TypeError(f'limit_number should have type int, but it has {type(limit_number)} '
                        f'and value {limit_number}')

    for i in iterator:
        single_anns = cocoGt_json[ANNOTATIONS][i]
        new_keypoints = CocoPreparator.take_default_skelet(single_anns)
        image_annot = find_image_annot(cocoGt_json, single_anns[IMAGE_ID])
        if image_annot is None:
            raise ModuleNotFoundError(f'Image id: {single_anns[IMAGE_ID]} was not found.')

        image_size = [image_annot[HEIGHT], image_annot[WIDTH]]

        # Scales for bbox and keypoints
        x_scale, y_scale = rescale_image(
            image_size=image_size,
            resize_to=[H, W],
            min_image_size=[min_h_size, min_w_size],
            use_force_resize=use_force_resize
        )
        scale_k = (x_scale, y_scale, 1)
        scale_bbox = (x_scale, y_scale)
        # Scale bbox and keypoints
        new_keypoints = (new_keypoints.reshape(-1, 3) * scale_k).reshape(-1).astype(np.float32).tolist()
        new_bbox = (np.array(single_anns[BBOX]).reshape(2, 2) * scale_bbox).reshape(-1).tolist()

        new_segmentation = single_anns[SEGMENTATION]
        # There is some garbage that stored in segmentation dict
        # Just skip it
        # TODO: Do something with this images
        if type(new_segmentation) == dict:
            continue

        # Scale segmentation points
        for m in range(len(new_segmentation)):
            single_new_seg_coord = np.array(new_segmentation[m]).reshape(-1, 2) * scale_bbox
            single_new_seg_coord = single_new_seg_coord.astype(np.float32).reshape(-1).tolist()
            new_segmentation[m] = single_new_seg_coord

        if mode_area_calculation == SEGMENTATION:
            # Calculate new value for 'area'
            new_area = float(np.sum(
                cocoGt.annToMask({
                    SEGMENTATION: new_segmentation,
                    IMAGE_ID: single_anns[IMAGE_ID]
                })
            ))
        elif mode_area_calculation == BBOX:
            new_area = float(new_bbox[2] * new_bbox[3])
        elif mode_area_calculation == KEYPOINTS:
            x = np.array(new_keypoints[0::3])
            y = np.array(new_keypoints[1::3])

            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            new_area = (x1 - x0) * (y1 - y0)
        else:
            raise TypeError(f'Unknown mode area type {mode_area_calculation}')

        # Fill our annotation with new information
        Maki_cocoGt_json[ANNOTATIONS].append(single_anns)
        Maki_cocoGt_json[ANNOTATIONS][i][KEYPOINTS] = new_keypoints
        Maki_cocoGt_json[ANNOTATIONS][i][BBOX] = new_bbox
        Maki_cocoGt_json[ANNOTATIONS][i][SEGMENTATION] = new_segmentation
        Maki_cocoGt_json[ANNOTATIONS][i][AREA] = new_area

        # Write img ids which we process
        if not used_ids[single_anns[IMAGE_ID]]:
            Maki_cocoGt_json[IMAGES].append(dict_id_by_image_info[single_anns[IMAGE_ID]])
            used_ids[single_anns[IMAGE_ID]] = True

    iterator.close()
    with open(path_to_save, 'w') as fp:
        json.dump(Maki_cocoGt_json, fp)


def find_image_annot(cocoGt_json: dict, img_id: int) -> dict:
    """
    Return annotation from `cocoGt_json` annotation according to `img_id`

    """
    for single_annot in cocoGt_json[IMAGES]:
        if single_annot[ID] == img_id:
            return single_annot

    return None

