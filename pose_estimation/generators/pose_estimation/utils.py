import tensorflow as tf

import numpy as np


def check_bounds(keypoints, image_size):
    check_min_x = tf.cast(keypoints[..., 1] > 0.0, dtype=tf.float32)
    check_min_y = tf.cast(keypoints[..., 0] > 0.0, dtype=tf.float32)

    check_min_xy = tf.expand_dims(check_min_x * check_min_y, axis=-1)

    check_x = tf.cast(keypoints[..., 1] < tf.cast(image_size[0], dtype=tf.float32), dtype=tf.float32)
    check_y = tf.cast(keypoints[..., 0] < tf.cast(image_size[1], dtype=tf.float32), dtype=tf.float32)

    check_xy = tf.expand_dims(check_x * check_y, axis=-1)

    return check_min_xy * check_xy


