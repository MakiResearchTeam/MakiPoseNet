import cv2
import numpy as np

from pose_estimation.model.utils.human import Human


def visualize_paf(
        img,
        pafs,
        colors=((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0))
):
    img = img.copy()
    color_iter = 0
    for i in range(pafs.shape[2]):
        paf_x = pafs[:, :, i, 0]
        paf_y = pafs[:, :, i, 1]
        len_paf = np.sqrt(paf_x ** 2 + paf_y ** 2)
        for x in range(0, img.shape[0], 8):
            for y in range(0, img.shape[1], 8):
                if len_paf[x,y] > 0.1:
                    img = cv2.arrowedLine(
                        img,
                        (y, x),
                        (int(y + 9 * paf_x[x, y]), int(x + 9 * paf_y[x, y])),
                        colors[color_iter],
                        1
                    )

                    color_iter += 1

                    if color_iter >= len(colors):
                        color_iter = 0
    return img


def draw_skeleton(image, humans: list, connect_indexes: list, color=(255, 0, 0)):
    for indx in range(len(humans)):
        human = humans[indx]
        print(f'is this human? in: {human}')
        if isinstance(human, Human):
            data = np.array(human.to_list()).reshape(-1, 3)
        else:
            print(np.array(human).shape)
            data = np.array(human).reshape(-1, 3)
        for j in range(len(connect_indexes)):
            single = connect_indexes[j]
            single_p1 = data[single[0]]
            single_p2 = data[single[1]]

            if single_p1[-1] > 1e-3 and single_p2[-1] > 1e-3:

                p_1 = (int(single_p1[0]), int(single_p1[1]))
                p_2 = (int(single_p2[0]), int(single_p2[1]))

                cv2.line(image, p_1, p_2, color=color, thickness=2)

    return image
