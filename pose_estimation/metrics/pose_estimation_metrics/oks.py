import numpy as np

# Constants, aka k in formule

BODY_K = [
    .026, .025, .025, .035, .035, .079, .079, .072, .072,
    .062, .062, 0.107, 0.107, .087, .087, .089, .089
]

FOOT_K = [
    0.068, 0.066, 0.066, 0.092, 0.094, 0.094
]

FACE_K = [
    0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025,
    0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043,
    0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012,
    0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007,
    0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011,
    0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011,
    0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008,
    0.007, 0.010, 0.008, 0.009, 0.009, 0.009, 0.007,
    0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008
]

LEFT_HAND_K = [
    0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
    0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 
    0.032, 0.02, 0.019, 0.022, 0.031
]

RIGHT_HAND_K = [
    0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 
    0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
    0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
]

ALL_K = BODY_K + FOOT_K + FACE_K + LEFT_HAND_K + RIGHT_HAND_K

# Constants for our skelet with 21 keypoints
MAKI_SKELET_K = [
    # body
    BODY_K[5],
    BODY_K[1],
    BODY_K[2],
    *BODY_K[5:],
    # hands
    LEFT_HAND_K[0],
    LEFT_HAND_K[-1],
    RIGHT_HAND_K[0],
    RIGHT_HAND_K[-1],
    # foot
    FOOT_K[0],
    FOOT_K[3]
]


def compute_oks(prediction: list, groundtruth: list, bboxes: list):
    """
    Compute OKS

    Parameters
    ----------
    prediction : list
        List contains of predicted keypoints.
        Each element has shape - (n_keypoints, 3)
    groundtruth : list
        List contains of groundtruth keypoints.
        Each element has shape - (n_keypoints, 3)
    bboxes : list
        BBoxes of human for each human in image,
        Each element has shape - (4)

    Returns
    -------

    """
    # prediction shape - (N, n_keypoints, 3)
    # ious shape - (N, n_keypoints)
    ious = np.zeros((len(prediction), len(prediction[0])))

    sigmas = np.array(MAKI_SKELET_K)
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for i, (pt, gt, gb) in enumerate(zip(prediction, groundtruth, bboxes)):
        # pt, gt shape - (n_keypoints, 3)
        # gb shape - (4)

        x0 = gb[0] - gb[2]
        x1 = gb[0] + gb[2] * 2
        y0 = gb[1] - gb[3]
        y1 = gb[1] + gb[3] * 2

        area = abs(x1 - x0) * abs(y1 - y0)
        # for each person on image
        for j, (pt_single, gt_single) in enumerate(zip(pt, gt)):
            # pt_single, gt_single shape - (3)
            d = np.array(pt_single)
            xd = d[0::3]
            yd = d[1::3]

            g = np.array(gt_single)
            xg = g[0:3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)

            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
            e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2
            if k1 > 0:
                e = e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious

