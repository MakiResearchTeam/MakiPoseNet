import numpy as np

from .utils import getHeadSize

KEYPOINTS = 'keypoints'


def computeSingleDist(gtSingleAnnots: list, dtSingleAnnots: list, distThresh=0.5, kpThresh=0.2):
    if len(dtSingleAnnots) == 0:
        # (gt_num, dt_num, kp_num) matrix
        dist_res = np.zeros((len(gtSingleAnnots), 1, 24)).astype(np.float32)
        correct_res = np.zeros((len(gtSingleAnnots), 1, 24)).astype(np.int32)
        return dist_res, correct_res

    # (gt_num, dt_num, kp_num) matrix
    dist_res = np.zeros((len(gtSingleAnnots), len(dtSingleAnnots), 24)).astype(np.float32)
    correct_res = np.zeros((len(gtSingleAnnots), len(dtSingleAnnots), 24)).astype(np.int32)

    # through each grouth truth
    for gt_annot_indx in range(len(gtSingleAnnots)):
        gt_single_annot = gtSingleAnnots[gt_annot_indx]

        # through each detected kp
        for dt_annot_indx in range(len(dtSingleAnnots)):
            dt_single_annot = dtSingleAnnots[dt_annot_indx]
            dt_keypoints = np.array(dt_single_annot[KEYPOINTS]).reshape(-1, 3)
            gt_keypoints = np.array(gt_single_annot[KEYPOINTS]).reshape(-1, 3)

            # each keypoint, measure distance
            for kp_indx in range(len(dt_keypoints)):
                correct = False
                gt_single_point = gt_keypoints[kp_indx]
                dt_single_point = dt_keypoints[kp_indx]

                gt_v = gt_single_point[-1]
                dt_v = dt_single_point[-1]
                gt_coords = np.array(gt_single_point)[:-1]
                dt_coords = np.array(dt_single_point)[:-1]

                if gt_v > 1e-3:
                    # compute distance between GT and prediction
                    d = np.linalg.norm(np.subtract(gt_coords, dt_coords))
                    # compute head size for distance normalization
                    headSize = getHeadSize(
                        gt_keypoints
                    )
                    if headSize * distThresh > d:
                        correct = True

                    # normalize distance
                    dNorm = d / headSize
                else:
                    if dt_v < kpThresh:
                        correct = True
                    dNorm = 1000 # np.inf
                dist_res[gt_annot_indx, dt_annot_indx, kp_indx] = dNorm
                correct_res[gt_annot_indx, dt_annot_indx, kp_indx] = int(correct)

    return dist_res, correct_res


def computePCKh_from_coco_annot(gt_coco_annot, dt_coco_annot, distThresh=None):
    if distThresh is None:
        distThresh = np.linspace(0, 0.5, 6).astype(np.float32)
    elif not isinstance(distThresh, list):
        distThresh = [distThresh]



    pass
