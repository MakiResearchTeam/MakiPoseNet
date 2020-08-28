import tensorflow as tf

import numpy as np

DEGREE2RAD = 180.0 / np.pi 


def check_bounds(keypoints, image_size):
    image_size = tf.convert_to_tensor(image_size)
    check_min_x = tf.cast(keypoints[..., 1] > 0.0, dtype=tf.float32)
    check_min_y = tf.cast(keypoints[..., 0] > 0.0, dtype=tf.float32)

    check_min_xy = tf.expand_dims(check_min_x * check_min_y, axis=-1)

    check_x = tf.cast(keypoints[..., 1] < tf.cast(image_size[0], dtype=tf.float32), dtype=tf.float32)
    check_y = tf.cast(keypoints[..., 0] < tf.cast(image_size[1], dtype=tf.float32), dtype=tf.float32)

    check_xy = tf.expand_dims(check_x * check_y, axis=-1)

    return check_min_xy * check_xy



def add_z_dim(x):
    """
    Add additional dimension filled in with ones

    """
    return tf.concat(
        [
            x,
            tf.ones_like(tf.expand_dims(tf.convert_to_tensor(x)[..., 0], axis=-1), dtype=tf.float32, optimize=False)
        ],
        axis=-1
    )


def get_rotate_matrix(image, angle):
    """
    Get rotation matrix for image with certain angle

    """
    shift_x = image.get_shape()[1].value // 2
    shift_y = image.get_shape()[0].value // 2

    shift_center = get_shift_matrix(-shift_x, -shift_y)

    shift_back = get_shift_matrix(shift_x, shift_y)

    rot_matrix = tf.stack([
        [       tf.math.cos(angle), tf.math.sin(angle), 0.0],
        [(-1) * tf.math.sin(angle), tf.math.cos(angle), 0.0],
        [           0.0,            0.0,                1.0],
    ])

    full_matrix = tf.matmul(tf.matmul(shift_back, rot_matrix), shift_center)

    return full_matrix

def get_rotate_matrix_batched(images, angle_batched):
    """
    Get rotation matrix for every image in the batch with certain angle in the angle_batched array

    """
    return tf.stack(
        [get_rotate_matrix(images[i], angle_batched[i]) for i in range(images.get_shape().as_list()[0])]
    )


def get_shift_matrix(dx, dy):
    """
    Get shift matrix with certain dx and dy shifts by certain axis (x and y)

    """
    
    return tf.stack([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-dx, -dy, 1.0],
    ])


def get_shift_matrix_batched(dx_batched, dy_batched):
    """
    Get batched shift matrix with certain dx and dy in dx_batched and dy_batched array

    """
    assert dx_batched.get_shape().as_list()[0] == dy_batched.get_shape().as_list()[0]

    return tf.stack(
        [get_shift_matrix(dx_batched[i], dy_batched[i]) for i in range(dx_batched.get_shape().as_list()[0])]
    )


def get_zoom_matrix(zoom):
    """
    Get zoom matrix with certain scale `zoom`

    """
    return tf.stack([
        [zoom, 0.0,  0.0],
        [0.0,  zoom, 0.0],
        [0.0,  0.0,  1.0]
    ])


def get_zoom_matrix_batched(zoom_batched):
    """
    Get batched zoom matrix for every scale in the `zoom_batched` array

    """
    return tf.stack(
        [get_zoom_matrix(zoom_batched[i]) for i in range(zoom_batched.get_shape().as_list()[0])]
    )


def apply_transformation(
        image,
        key_points,
        use_rotation=False,
        angle=None,
        use_shift=False,
        dx=None,
        dy=None,
        use_zoom=False,
        zoom_scale=None
):
    """
    Apply transformation to an image and keypoints

    Returns
    -------
    tf.Tensor
        Batch of the transformed image
    tf.Tensor
        Batch of the transformed keypoints

    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    zoom_matrix = None
    shift_matrix = None
    rotation_matrix = None

    if use_zoom and zoom_scale is not None:
        zoom_scale = tf.convert_to_tensor(zoom_scale, dtype=tf.float32)
        zoom_matrix = get_zoom_matrix(zoom_scale)

    if use_shift and dx is not None and dy is not None:
        dx = tf.convert_to_tensor(dx, dtype=tf.float32)
        dy = tf.convert_to_tensor(dy, dtype=tf.float32)
        shift_matrix = get_shift_matrix(dx, dy)

    if use_rotation and angle is not None:
        angle = tf.convert_to_tensor(angle, dtype=tf.float32) / DEGREE2RAD
        rotation_matrix = get_rotate_matrix(image, angle)

    kp = tf.convert_to_tensor(key_points, dtype=tf.float32)
    kp = add_z_dim(kp)

    full_matrix = tf.ones([3, 3], dtype=tf.float32)
    use_ones = False

    if rotation_matrix is not None:
        full_matrix = tf.multiply(full_matrix, rotation_matrix)
        use_ones = True

    if shift_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, shift_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, shift_matrix)
        use_ones = True

    if zoom_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, zoom_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, zoom_matrix)
        use_ones = True

    proj_matrix = tf.contrib.image.matrices_to_flat_transforms(tf.transpose(full_matrix))

    transformed_image = tf.contrib.image.transform([image], proj_matrix)[0]
    transformed_kp = tf.matmul(kp, tf.linalg.inv(full_matrix))

    return transformed_image, transformed_kp[..., :2]




def apply_transformation_batched(
        images,
        key_points_batched,
        use_rotation=False,
        angle_batched=None,
        use_shift=False,
        dx_batched=None,
        dy_batched=None,
        use_zoom=False,
        zoom_scale_batched=None
):
    """
    Apply transformation to images and keypoints in certain batch (i.e. batch size)

    Returns
    -------
    tf.Tensor
        Batch of transformed images
    tf.Tensor
        Batch of transformed keypoints

    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    N = images.get_shape().as_list()[0]

    zoom_matrix = None
    shift_matrix = None
    rotation_matrix = None

    if use_zoom and zoom_scale_batched is not None:
        zoom_scale_batched = tf.convert_to_tensor(zoom_scale_batched, dtype=tf.float32)
        zoom_matrix = get_zoom_matrix_batched(zoom_scale_batched)

    if use_shift and dx_batched is not None and dy_batched is not None:
        dx_batched = tf.convert_to_tensor(dx_batched, dtype=tf.float32)
        dy_batched = tf.convert_to_tensor(dy_batched, dtype=tf.float32)
        shift_matrix = get_shift_matrix_batched(dx_batched, dy_batched)

    if use_rotation and angle_batched is not None:
        angle_batched = tf.convert_to_tensor(angle_batched, dtype=tf.float32) / DEGREE2RAD
        rotation_matrix = get_rotate_matrix_batched(images, angle_batched)

    kp = tf.convert_to_tensor(key_points_batched, dtype=tf.float32)
    kp = add_z_dim(kp)

    full_matrix = tf.ones([N, 3, 3], dtype=tf.float32)
    use_ones = False

    if rotation_matrix is not None:
        full_matrix = tf.multiply(full_matrix, rotation_matrix)
        use_ones = True

    if shift_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, shift_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, shift_matrix)
        use_ones = True

    if zoom_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, zoom_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, zoom_matrix)
        use_ones = True

    proj_matrix = tf.contrib.image.matrices_to_flat_transforms(
        [
            tf.transpose(full_matrix[j])
            for j in range(N)
        ]
    )

    transformed_image = tf.contrib.image.transform(images, proj_matrix)
    transformed_kp = tf.stack(
        [
            tf.matmul(kp[j], tf.linalg.inv(full_matrix[j]))
            for j in range(N)
        ]
    )

    return transformed_image, transformed_kp[..., :2]
