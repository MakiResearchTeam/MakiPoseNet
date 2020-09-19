from .tester import Tester
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.visualize_tools import visualize_paf


class CocoTester(Tester):
    TEST_IMAGE = 'test_image'
    TEST_N = "test_{}"

    _EXCEPTION_IMAGE_WAS_NOT_FOUND = "Image by path {0} was not found!"

    HEATMAP_CENTER_BODY_IMAGE = 'heatmap_center_body_image'
    HEATMAP_CENTER_BODY_TEXT = 'heat map center body'
    HEATMAP_CENTER_BODY_INDEX = 0

    HEATMAP_LEFT_SHOULDER_IMAGE = 'heatmap_left_shoulder_image'
    HEATMAP_LEFT_SHOULDER_TEXT = "heatmap left shoulder"
    HEATMAP_LEFT_SHOULDER_INDEX = 6

    HEATMAP_RIGHT_SHOULDER_IMAGE = 'heatmap_right_shoulder_image'
    HEATMAP_RIGHT_SHOULDER_TEXT = "heatmap right shoulder"
    HEATMAP_RIGHT_SHOULDER_INDEX = 7

    HEATMAP_RIGHT_HAND_IMAGE = 'heatmap_right_hand_image'
    HEATMAP_RIGHT_HAND_TEXT = "heatmap right hand"
    HEATMAP_RIGHT_HAND_INDEX = 11

    HEATMAP_LEFT_HAND_IMAGE = 'heatmap_left_hand_image'
    HEATMAP_LEFT_HAND_TEXT = "heatmap left hand"
    HEATMAP_LEFT_HAND_INDEX = 10

    DRAW_LIST = [
        [HEATMAP_CENTER_BODY_TEXT, HEATMAP_CENTER_BODY_INDEX],
        [HEATMAP_LEFT_SHOULDER_TEXT, HEATMAP_LEFT_SHOULDER_INDEX],
        [HEATMAP_RIGHT_SHOULDER_TEXT, HEATMAP_RIGHT_SHOULDER_INDEX],
        [HEATMAP_RIGHT_HAND_TEXT, HEATMAP_RIGHT_HAND_INDEX],
        [HEATMAP_LEFT_HAND_TEXT, HEATMAP_LEFT_HAND_INDEX]
    ]

    PAFF_IMAGE = 'paff_image'
    ITERATION_COUNTER = 'iteration_counter'

    def _init(self, config, normalization_method=None):

        if type(self._config[CocoTester.TEST_IMAGE]) is not list:
            test_images_path = [self._config[CocoTester.TEST_IMAGE]]
        else:
            test_images_path = self._config[CocoTester.TEST_IMAGE]

        self._norm_images = []
        self._test_images = []
        self._names = []

        for i, single_path in enumerate(test_images_path):
            test_image = cv2.imread(single_path)
            if test_image is None:
                raise TypeError(CocoTester._EXCEPTION_IMAGE_WAS_NOT_FOUND.format(self._config[CocoTester.TEST_IMAGE]))

            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            im_shape = test_image.shape

            if normalization_method is not None:
                self._norm_images.append(normalization_method(test_image).reshape(1, *im_shape).astype(np.float32))
            else:
                self._norm_images.append(((test_image.reshape(1, *im_shape) - 127.5) / 127.5).astype(np.float32))

            # The image has to have batch dimension
            self._test_images.append(test_image.reshape(1, *im_shape).astype(np.uint8))
            self._names.append(CocoTester.TEST_N.format(i))
            self.add_image(self._names[-1], n_images=7)

        self.add_scalar(CocoTester.ITERATION_COUNTER)

    def evaluate(self, model, iteration):

        dict_images = {CocoTester.ITERATION_COUNTER: iteration}

        for i, (single_norm, single_test) in enumerate(zip(self._norm_images, self._test_images)):
            single_batch = [single_test]
            peaks, heatmap, paf = model.predict(
                np.concatenate([single_norm] * model.get_batch_size(), axis=0),
                using_estimate_alg=False
            )

            drawed_paff = np.expand_dims(visualize_paf(single_test, paf[0]), axis=0)

            single_batch.append(drawed_paff)
            for single_draw_params in CocoTester.DRAW_LIST:
                index = single_draw_params[1]
                name_p = single_draw_params[0]

                single_batch.append(
                    np.expand_dims(
                        self.draw_heatmap(
                            heatmap[0][..., index],
                            name_p
                        ),
                        axis=0
                    )
                )
            dict_images.update({self._names[i]: np.concatenate(single_batch, axis=0).astype(np.uint8)})

        # Do here the fucking evaluation
        self.write_summaries(
            summaries=dict_images,
            step=iteration
        )

    def draw_heatmap(self, heatmap, name_heatmap, shift_image=60):
        dpi = 80
        h,w = heatmap.shape

        figsuze = w / float(dpi), h / float(dpi)

        fig = plt.figure(frameon=False, figsize=figsuze, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        sns.heatmap(heatmap)
        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = np.reshape(data, (h, w, 3))
        img = np.ones((h + shift_image, w, 3)) * 255.0
        img[:h, :w] = data

        cv2.putText(
            img,
            name_heatmap,
            (shift_image // 4, h + shift_image // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            2
        )

        return img.astype(np.uint8)
