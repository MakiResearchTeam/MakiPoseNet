import cv2
from pose_estimation.data_preparation import CONNECT_INDEXES
<<<<<<< HEAD
from pose_estimation.utils.visualize_tools import draw_skeleton


class SkeletonDrawer:
    def __init__(self, video_path, connect_indexes=CONNECT_INDEXES, fps=20, color=(255, 0, 0)):
        self._video_path = video_path
        self._connect_indexes = connect_indexes
        self._fps = fps
        self._color = color
=======


class SkeletonDrawer:
    def __init__(self, video_path, connect_indexes=CONNECT_INDEXES, fps=20):
        self._video_path = video_path
        self._connect_indexes = connect_indexes
        self._fps = fps
>>>>>>> origin/add_new_pafs
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
<<<<<<< HEAD
            image = draw_skeleton(image, prediction, self._connect_indexes, self._color)
            self._video.write(image)

=======
            self.draw_skeleton(image, prediction)
            self._video.write(image)

    def draw_skeleton(self, image, prediction):
        for human in prediction:
            self.draw_human(image, human)

    def draw_human(self, image, human):
        data = human.to_dict()
        for j in range(len(self._connect_indexes)):
            single = self._connect_indexes[j]
            if data.get(str(single[0])) is not None and data.get(str(single[1])) is not None and \
                    data.get(str(single[0]))[0] > 0 and data.get(str(single[1]))[1] > 0:
                p_1 = (int(data.get(str(single[0]))[0]), int(data.get(str(single[0]))[1]))
                p_2 = (int(data.get(str(single[1]))[0]), int(data.get(str(single[1]))[1]))

                cv2.line(image, p_1, p_2, color=(255, 0, 0), thickness=2)

>>>>>>> origin/add_new_pafs
    def release(self):
        self._video.release()
