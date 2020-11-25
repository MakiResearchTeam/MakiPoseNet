import cv2
from .constants import CONNECT_INDEXES
from pose_estimation.utils.visual_tools.visualize_tools import draw_skeleton


class SkeletonDrawer:
    def __init__(self, video_path, connect_indexes=CONNECT_INDEXES, fps=20, color=(255, 0, 0)):
        self._video_path = video_path
        self._connect_indexes = connect_indexes
        self._fps = fps
        self._color = color
        self._video = None

    def _init(self, frame_size):
        height, width = frame_size
        self._video = cv2.VideoWriter(
            self._video_path, cv2.VideoWriter_fourcc(*'mp4v'), self._fps,
            (width, height))

    def write(self, images, predictions):
        """
        Draws skeletons on the `images` according to the give `predictions`.

        Parameters
        ----------
        images : list
            List of images (ndarrays).
        predictions : list
            List of lists that contain instances of class Human.
        """
        if self._video is None:
            h, w, c = images[0].shape
            self._init((h, w))

        for image, prediction in zip(images, predictions):
            image = draw_skeleton(image, prediction, self._connect_indexes, self._color)
            self._video.write(image)

    def release(self):
        self._video.release()
