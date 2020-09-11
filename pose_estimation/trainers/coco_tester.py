from .tester import Tester
import cv2
import numpy as np


class CocoTester(Tester):
    TEST_IMAGE = 'test_image'
    ITERATION_COUNTER = 'iteration_counter'

    def _init(self, config, normalization_method=None):
        test_image = cv2.imread(self._config[CocoTester.TEST_IMAGE])
        im_shape = test_image.shape

        if normalization_method is not None:
            test_image = normalization_method(test_image)

        # The image has to have batch dimension
        self._test_image = test_image.reshape(1, *im_shape).astype(np.float32)
        self.add_image(CocoTester.TEST_IMAGE)

        self.add_scalar(CocoTester.ITERATION_COUNTER)

    def evaluate(self, model, iteration):
        peaks, heatmap, paf = model.predict(
            np.concatenate([self._test_image] * model.get_batch_size(), axis=0),
            using_estimate_alg=False
        )

        # Do here the fucking evaluation
        self.write_summaries(
            summaries={
                CocoTester.TEST_IMAGE: heatmap[0][..., 0],
                CocoTester.ITERATION_COUNTER: iteration
            },
            step=iteration
        )
