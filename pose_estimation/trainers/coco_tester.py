from .tester import Tester
import cv2
import numpy as np
from ..utils.visualize_tools import visualize_paf


class CocoTester(Tester):
    TEST_IMAGE = 'test_image'

    HEATMAP_CENTER_BODY_IMAGE = 'heatmap_center_body_image'
    HEATMAP_LEFT_SHOULDER_IMAGE = 'heatmap_left_shoulder_image'
    HEATMAP_RIGHT_SHOULDER_IMAGE = 'heatmap_right_shoulder_image'
    HEATMAP_RIGHT_HAND_IMAGE = 'heatmap_right_hand_image'
    HEATMAP_LEFT_HAND_IMAGE = 'heatmap_left_hand_image'

    PAFF_IMAGE = 'paff_image'
    ITERATION_COUNTER = 'iteration_counter'

    def _init(self, config, normalization_method=None):
        test_image = cv2.imread(self._config[CocoTester.TEST_IMAGE])
        im_shape = test_image.shape

        if normalization_method is not None:
            self._norm_image = normalization_method(test_image).reshape(1, *im_shape).astype(np.float32)
        else:
            self._norm_image = test_image.reshape(1, *im_shape).astype(np.float32)

        # The image has to have batch dimension
        self._test_image = test_image.reshape(1, *im_shape).astype(np.uint8)
        self.add_image(CocoTester.TEST_IMAGE)
        self.add_image(CocoTester.PAFF_IMAGE)

        self.add_image(CocoTester.HEATMAP_CENTER_BODY_IMAGE)
        self.add_image(CocoTester.HEATMAP_LEFT_SHOULDER_IMAGE)
        self.add_image(CocoTester.HEATMAP_RIGHT_SHOULDER_IMAGE)
        self.add_image(CocoTester.HEATMAP_RIGHT_HAND_IMAGE)
        self.add_image(CocoTester.HEATMAP_LEFT_HAND_IMAGE)

        self.add_scalar(CocoTester.ITERATION_COUNTER)

    def evaluate(self, model, iteration):
        peaks, heatmap, paf = model.predict(
            np.concatenate([self._norm_image] * model.get_batch_size(), axis=0),
            using_estimate_alg=False
        )

        drawed_paff = np.expand_dims(visualize_paf(self._test_image[0], paf[0]), axis=0)

        # Do here the fucking evaluation
        self.write_summaries(
            summaries={
                CocoTester.HEATMAP_CENTER_BODY_IMAGE: (heatmap[:1][..., :1] * 255.0).astype(np.uint8),
                CocoTester.HEATMAP_LEFT_SHOULDER_IMAGE: (heatmap[:1][..., 6:7] * 255.0).astype(np.uint8),
                CocoTester.HEATMAP_RIGHT_SHOULDER_IMAGE: (heatmap[:1][..., 7:8] * 255.0).astype(np.uint8),
                CocoTester.HEATMAP_RIGHT_HAND_IMAGE: (heatmap[:1][..., 11:12] * 255.0).astype(np.uint8),
                CocoTester.HEATMAP_LEFT_HAND_IMAGE: (heatmap[:1][..., 10:11] * 255.0).astype(np.uint8),
                CocoTester.PAFF_IMAGE: drawed_paff.astype(np.uint8),
                CocoTester.TEST_IMAGE: self._test_image,
                CocoTester.ITERATION_COUNTER: iteration
            },
            step=iteration
        )
