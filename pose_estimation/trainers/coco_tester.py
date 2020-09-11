from .tester import Tester
import cv2


class CocoTester(Tester):
    TEST_IMAGE = 'test_image'

    def _init(self, config):
        test_image = cv2.imread(self._config[CocoTester.TEST_IMAGE])
        im_shape = test_image.shape
        # The image has to have batch dimension
        self._test_image = test_image.reshape(1, *im_shape)
        self.add_image(CocoTester.TEST_IMAGE)

    def evaluate(self, model, iteration):
        self.write_summaries(
            {
                CocoTester.TEST_IMAGE: self._test_image
            }
        )
