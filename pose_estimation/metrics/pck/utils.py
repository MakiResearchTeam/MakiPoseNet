import numpy as np


def getHeadSize(dt_keypoints):
    if dt_keypoints[1][-1] > 1e-3 and (dt_keypoints[2][-1] > 1e-3 or dt_keypoints[3][-1] > 1e-3):
        counter = 0
        avg_point = np.zeros(2).astype(np.float32)
        if dt_keypoints[2][-1] > 1e-3:
            avg_point += dt_keypoints[2][:2]
            counter += 1

        if dt_keypoints[3][-1] > 1e-3:
            avg_point += dt_keypoints[3][:2]
            counter += 1

        avg_point /= counter
        one_point = dt_keypoints[1][:2]

        headSize = 2.0 * np.linalg.norm(np.subtract(avg_point, one_point))
    else:
        x_1_max = np.max(dt_keypoints[:, 0])
        x_2_min = np.min(dt_keypoints[:, 0])

        y_1_max = np.max(dt_keypoints[:, 1])
        y_2_min = np.min(dt_keypoints[:, 1])

        headSize = np.linalg.norm(np.subtract([x_1_max, y_1_max], [x_2_min, y_2_min])) / 16.0  # 12.0
    return headSize
