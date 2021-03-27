# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
import os

from makiflow.generators.pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from .data_preparation import (IMAGE_FNAME, KEYPOINTS_FNAME,
                               KEYPOINTS_MASK_FNAME, IMAGE_PROPERTIES_FNAME,
                               ABSENT_HUMAN_MASK_FNAME, ALPHA_MASK_FNAME)
from .utils import check_bounds, apply_transformation, apply_transformation_batched, cutout_kp_in_box
from pose_estimation.utils.nns_tools.preprocess import preprocess_input

import tensorflow as tf
import numpy as np
import glob


class RIterator:
    IMAGE = 'IMAGE'
    ABSENT_HUMAN_MASK = 'ABSENT_HUMAN_MASK'
    KEYPOINTS = 'KEYPOINTS'
    KEYPOINTS_MASK = 'KEYPOINTS_MASK'
    IMAGE_PROPERTIES = 'IMAGE_PROPERTIES'
    HEATMAP = 'HEATMAP'
    ALPHA_MASK = 'ALPHA_MASK'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            shape_keypoints: list,
            shape_image_properties=[3],
            image_dtype=tf.float32,
            keypoints_dtype=tf.float32,
            keypoints_mask_dtype=tf.float32,
            image_properties_dtype=tf.float32,
    ):
        """
        Method to load data from records

        Parameters
        ----------
        shape_keypoints : list
            Shape of keypoints in tfrecords [number_of_keypoints, n_people]
            For example: [24, 2]
        shape_image_properties : list
            Shape of image properties in tfrecords,
            By default equal to [3], where 0th - Height, 1th - Width, 2th - number of channels of the image
        image_dtype : tf.dtype
            Type of the image in tfrecords
        keypoints_dtype : tf.dtype
            Type of keypoints in tfrecords
        keypoints_mask_dtype : tf.dtype
            Type of the keypoints mask in tfrecords
        image_properties_dtype : tf.dtype
            Type of image properties in tfrecords
        """
        self.shape_keypoints = shape_keypoints + [2]
        self.shape_image_properties = shape_image_properties
        self.shape_keypoints_mask = shape_keypoints + [1]

        self.image_dtype = image_dtype
        self.keypoints_dtype = keypoints_dtype
        self.keypoints_mask_dtype = keypoints_mask_dtype
        self.image_properties_dtype = image_properties_dtype

    def read_record(self, serialized_example):
        r_feature_description = {
            IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
            ABSENT_HUMAN_MASK_FNAME: tf.io.FixedLenFeature((), tf.string),
            KEYPOINTS_FNAME: tf.io.FixedLenFeature((), tf.string),
            KEYPOINTS_MASK_FNAME: tf.io.FixedLenFeature((), tf.string),
            IMAGE_PROPERTIES_FNAME: tf.io.FixedLenFeature((), tf.string),
            ALPHA_MASK_FNAME: tf.io.FixedLenFeature((), tf.string)
        }

        example = tf.io.parse_single_example(serialized_example, r_feature_description)

        # Extract the data from the example
        image_tensor = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        image_mask = tf.io.parse_tensor(example[ABSENT_HUMAN_MASK_FNAME], out_type=self.image_dtype)
        keypoints_tensor = tf.io.parse_tensor(example[KEYPOINTS_FNAME], out_type=self.keypoints_dtype)

        keypoints_mask_tensor = tf.io.parse_tensor(example[KEYPOINTS_MASK_FNAME], out_type=self.keypoints_mask_dtype)
        image_properties_tensor = tf.io.parse_tensor(example[IMAGE_PROPERTIES_FNAME],
                                                     out_type=self.image_properties_dtype)
        alpha_mask_tensor = tf.io.parse_tensor(example[ALPHA_MASK_FNAME], out_type=tf.uint8)
        # Give the data its shape because it doesn't have it right after being extracted
        keypoints_tensor.set_shape(self.shape_keypoints)
        keypoints_mask_tensor.set_shape(self.shape_keypoints_mask)
        image_properties_tensor.set_shape(self.shape_image_properties)

        output_dict = {
            RIterator.IMAGE: image_tensor,
            RIterator.ABSENT_HUMAN_MASK: image_mask,
            RIterator.KEYPOINTS: keypoints_tensor,
            RIterator.KEYPOINTS_MASK: keypoints_mask_tensor,
            RIterator.IMAGE_PROPERTIES: image_properties_tensor,
            RIterator.ALPHA_MASK: alpha_mask_tensor
        }

        return output_dict


class RandomCropMethod(TFRPostMapMethod):

    def __init__(self, crop_h: int, crop_w: int, image_last_dimension=3):
        """
        Perform random crop of the input images and their corresponding uvmaps.
        Parameters
        ----------
        crop_h : int
            Height of the crop.
        crop_w : int
            Width of the crop.
        image_last_dimension : int
            Number of channels of images, by default equal to 3

        """
        super().__init__()
        self._crop_w = crop_w
        self._crop_h = crop_h
        self._image_crop_size = [crop_h, crop_w, image_last_dimension]
        self._image_mask_crop_size = [crop_h, crop_w, 1]

        self._image_crop_size_tf = tf.constant(np.array([crop_h, crop_w, image_last_dimension], dtype=np.int32))
        self._image_mask_crop_size_tf  = tf.constant(np.array([crop_h, crop_w, 1], dtype=np.int32))

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        image = element[RIterator.IMAGE]
        image_mask = element[RIterator.ABSENT_HUMAN_MASK]
        keypoints = element[RIterator.KEYPOINTS]
        keypoints_mask = element[RIterator.KEYPOINTS_MASK]
        alpha_mask = element[RIterator.ALPHA_MASK]
        # This is an adapted code from the original TensorFlow's `random_crop` method
        limit = tf.shape(image) - self._image_crop_size_tf + 1
        offset = tf.random_uniform(
            shape=[3],
            dtype=tf.int32,
            # it is unlikely that a tensor with shape more than 10000 will appear
            maxval=10000
        ) % limit

        cropped_image = tf.slice(image, offset, self._image_crop_size_tf)
        cropped_image_mask = tf.slice(image_mask, offset, self._image_mask_crop_size_tf)
        #cropped_alpha_mask = tf.slice(alpha_mask, offset, self._image_mask_crop_size_tf)
        cropped_keypoints = keypoints - tf.cast(tf.stack([offset[1], offset[0]]), dtype=tf.float32)
        # After slicing the tensors doesn't have proper shape. They get instead [None, None, None].
        # We can't use tf.Tensors for setting shape because they are note iterable what causes errors.
        cropped_image.set_shape(self._image_crop_size)
        cropped_image_mask.set_shape(self._image_mask_crop_size)
        #cropped_alpha_mask.set_shape(self._image_mask_crop_size)
        # Check which keypoints are beyond the image
        correct_keypoints_mask = keypoints_mask * check_bounds(cropped_keypoints, self._image_crop_size_tf)

        element[RIterator.IMAGE] = cropped_image
        element[RIterator.KEYPOINTS] = cropped_keypoints
        element[RIterator.ABSENT_HUMAN_MASK] = cropped_image_mask
        element[RIterator.KEYPOINTS_MASK] = correct_keypoints_mask
        #element[RIterator.ALPHA_MASK] = cropped_alpha_mask
        return element


class AugmentationPostMethod(TFRPostMapMethod):

    def __init__(self,
                 use_rotation=True,
                 angle_min=-30.0,
                 angle_max=30.0,
                 use_shift=False,
                 dx_min=None,
                 dx_max=None,
                 dy_min=None,
                 dy_max=None,
                 use_zoom=True,
                 zoom_min=0.9,
                 zoom_max=1.1
                 ):
        """
        Perform augmentation of images (rotation, shift, zoom)

        Parameters
        ----------
        use_rotation : bool
            If equal to True, will be performed rotation to image
        angle_min : float
            Minimum angle of the random rotation
        angle_max : float
            Maximum angle of the random rotation
        use_shift : bool
            If equal to True, will be performed shift to image
        dx_min : float
            Minimum shift by x axis of the random shift
        dx_max : float
            Maximum shift by x axis of the random shift
        dy_min : float
            Minimum shift by y axis of the random shift
        dy_max : float
            Maximum shift by y axis of the random shift
        use_zoom : bool
            If equal to True, will be performed zoom to image
        zoom_min : float
            Minimum zoom coeff of the random zoom
        zoom_max : float
            Maximum zoom of the random zoom

        """
        super().__init__()
        self.use_rotation = use_rotation
        if use_rotation and (angle_max is None or angle_min is None):
            raise ValueError(
                'Parameters angle_max and angle_min are should be not None values' + \
                'If `use_rotation` equal to True'
            )

        if use_rotation and angle_max < angle_min:
            raise ValueError(
                'Parameter angle_max should be bigger that angle_min, but ' + \
                f'angle_max = {angle_max} and angle_min = {angle_min} were given'
            )

        self.angle_min = angle_min
        self.angle_max = angle_max

        self.use_shift = use_shift
        if use_shift and (dx_min is None or dx_max is None or dy_min is None or dy_max is None):
            raise ValueError(
                'Parameters dx_min, dx_max, dy_min and dy_max are should be not None values' + \
                'If use_shift equal to True'
            )

        if use_shift and dx_max < dx_min:
            raise ValueError(
                'Parameter dx_max should be bigger that dx_min, but ' + \
                f'dx_max = {dx_max} and dx_min = {dx_min} were given'
            )

        self.dx_min = dx_min
        self.dx_max = dx_max

        if use_shift and dy_max < dy_min:
            raise ValueError(
                'Parameter dy_max should be bigger that dy_min, but ' + \
                f'dy_max = {dy_max} and dy_min = {dy_min} were given'
            )

        self.dy_min = dy_min
        self.dy_max = dy_max

        self.use_zoom = use_zoom
        if use_zoom and (zoom_min is None or zoom_max is None):
            raise ValueError(
                'Parameters zoom_min and zoom_max are should be not None values' + \
                'If use_zoom equal to True'
            )

        if use_shift and zoom_max < zoom_min:
            raise ValueError(
                'Parameter zoom_max should be bigger that zoom_min, but ' + \
                f'zoom_max = {zoom_max} and zoom_min = {zoom_min} were given'
            )

        self.zoom_min = zoom_min
        self.zoom_max = zoom_max

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example

        if not self.use_shift and not self.use_zoom and not self.use_rotation:
            return element

        image = element[RIterator.IMAGE]
        image_mask = element[RIterator.ABSENT_HUMAN_MASK]
        keypoints = element[RIterator.KEYPOINTS]
        keypoints_mask = element[RIterator.KEYPOINTS_MASK]

        image_shape = image.get_shape().as_list()
        angle = None
        dy = None
        dx = None
        zoom = None

        if len(image_shape) == 3:
            if self.use_rotation:
                angle = tf.random.uniform([], minval=self.angle_min, maxval=self.angle_max, dtype='float32')

            if self.use_shift:
                dy = tf.random.uniform([], minval=self.dy_min, maxval=self.dy_max, dtype='float32')
                dx = tf.random.uniform([], minval=self.dx_min, maxval=self.dx_max, dtype='float32')

            if self.use_zoom:
                zoom = tf.random.uniform([], minval=self.zoom_min, maxval=self.zoom_max, dtype='float32')

            transformed_image_and_mask, transformed_keypoints = apply_transformation(
                [image, image_mask],
                keypoints,
                use_rotation=self.use_rotation,
                angle=angle,
                use_shift=self.use_shift,
                dx=dx,
                dy=dy,
                use_zoom=self.use_zoom,
                zoom_scale=zoom
            )
            transformed_image = transformed_image_and_mask[0]
            transformed_image_mask = transformed_image_and_mask[1]
            # Check which keypoints are beyond the image
            correct_keypoints_mask = keypoints_mask * check_bounds(transformed_keypoints, image_shape)
        else:
            # Batched
            N = image_shape[0]
            if self.use_rotation:
                angle = tf.random.uniform([N], minval=self.angle_min, maxval=self.angle_max, dtype='float32')

            if self.use_shift:
                dy = tf.random.uniform([N], minval=self.dy_min, maxval=self.dy_max, dtype='float32')
                dx = tf.random.uniform([N], minval=self.dx_min, maxval=self.dx_max, dtype='float32')

            if self.use_zoom:
                zoom = tf.random.uniform([N], minval=self.zoom_min, maxval=self.zoom_max, dtype='float32')

            transformed_image_and_mask, transformed_keypoints = apply_transformation_batched(
                [image, image_mask],
                keypoints,
                use_rotation=self.use_rotation,
                angle_batched=angle,
                use_shift=self.use_shift,
                dx_batched=dx,
                dy_batched=dy,
                use_zoom=self.use_zoom,
                zoom_scale_batched=zoom
            )
            transformed_image = transformed_image_and_mask[0]
            transformed_image_mask = transformed_image_and_mask[1]
            # Check which keypoints are beyond the image
            correct_keypoints_mask = keypoints_mask * check_bounds(transformed_keypoints, image_shape[1:])

        element[RIterator.IMAGE] = transformed_image
        element[RIterator.ABSENT_HUMAN_MASK] = transformed_image_mask
        element[RIterator.KEYPOINTS] = transformed_keypoints
        element[RIterator.KEYPOINTS_MASK] = correct_keypoints_mask
        return element


class NormalizePostMethod(TFRPostMapMethod):

    def __init__(self,
                 mode='tf',
                 divider=None,
                 shift=None,
                 use_float64=True):
        """
        Normalizes the tensor by dividing it by the `divider`.
        Parameters
        ----------
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                  then will zero-center each color channel with
                  respect to the ImageNet dataset,
                  without scaling.
            - tf: will scale pixels between -1 and 1,
                  sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                  will normalize each channel with respect to the
                  ImageNet dataset.
            If equal to None, will be used `divider` and `shift` variables

        divider : float or int
            The number to divide the tensor by,
            Can be equal to None, i.e. will be not used,
            For example, `x` is input tensor
            Output: `x` * `divider`
        shift : float
            The number to shift the tensor after divide operation,
            Can be equal to None, i.e. will be not used,
            For example, x is input tensor
            Output: `x` - `shift`
        use_float64 : bool
            Set to True if you want the tensor to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        """
        super().__init__()
        self.use_float64 = use_float64
        self.mode = mode

        if mode is None:
            if divider is None:
                divider = 1.0

            if shift is None:
                shift = 0.0

            if use_float64:
                self.divider = tf.constant(divider, dtype=tf.float64)
                self.shift = tf.constant(shift, dtype=tf.float64)
            else:
                self.divider = tf.constant(divider, dtype=tf.float32)
                self.shift = tf.constant(shift, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example

        image_tensor = element[RIterator.IMAGE]
        if self.mode is None:
            if self.use_float64:
                image_tensor = tf.cast(image_tensor, dtype=tf.float64)
                image_tensor = image_tensor / self.divider - self.shift
                image_tensor = tf.cast(image_tensor, dtype=tf.float32)
            else:
                image_tensor = image_tensor / self.divider - self.shift
        else:
            if self.use_float64:
                image_tensor = tf.cast(image_tensor, dtype=tf.float64)
                image_tensor = preprocess_input(image_tensor, mode=self.mode)
                image_tensor = tf.cast(image_tensor, dtype=tf.float32)
            else:
                image_tensor = preprocess_input(image_tensor, mode=self.mode)
        element[RIterator.IMAGE] = image_tensor

        return element


class RGB2BGRPostMethod(TFRPostMapMethod):
    RGB2BGR_IMAGE = 'BGR2RGB_input'

    def __init__(self):
        """
        Used for swapping color channels in tensors from RGB to BGR format.

        """
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]
        # Swap channels
        element[RIterator.IMAGE] = tf.reverse(image, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_IMAGE)
        return element


class BinaryHeatmapMethod(TFRPostMapMethod):
    def __init__(self, im_size, radius, map_dtype=tf.int32):
        """
        Generates hard keypoint maps using highly optimized vectorization. May cause OOM error due to high
        memory consumption.
        Parameters
        ----------
        im_size : 2d tuple
            Contains width and height (w, h) of the image for which to generate the map.
        radius : int
            Radius of a label-circle around the keypoint.
        map_dtype : tf.dtype
            Dtype of the generated map. Use tf.int32 for binary classification and tf.float32 for
            regression.
        """
        super().__init__()
        self.im_size = im_size
        self.radius = tf.convert_to_tensor(radius, dtype=tf.float32)
        self.map_dtype = map_dtype
        # Prepare the grid.
        x_grid, y_grid = np.meshgrid(np.arange(im_size[0]), np.arange(im_size[1]))
        xy_grid = np.stack([x_grid, y_grid], axis=-1)
        self.xy_grid = tf.convert_to_tensor(xy_grid, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        keypoints = element[RIterator.KEYPOINTS]
        # keypoints = tf.transpose(keypoints, perm=[1, 0, 2])[:, :68]

        maps = self.__build_heatmap_batch(keypoints, self.xy_grid, self.radius)
        element[RIterator.HEATMAP] = maps
        return element

    def __build_heatmap_batch(self, kp, xy_grid, radius):
        # Build maps for keypoints of the same class for multiple people
        # and then aggregate generated maps.
        # [h, w]
        fn_p = lambda kp, xy_grid, radius: tf.reduce_max(
            BinaryHeatmapMethod.__build_heatmap_mp(
                kp,
                xy_grid,
                radius,
                destination_call=self.__build_heatmap
            ),
            axis=0
        )
        # Build maps for keypoints of multiple classes.
        # [c, h, w]
        fn_c = lambda kp, xy_grid, radius: BinaryHeatmapMethod.__build_heatmap_mp(
            kp,
            xy_grid,
            radius,
            destination_call=fn_p
        )
        # Build a batch of maps.
        # [b, c, h, w]
        fn_b = lambda kp, xy_grid, radius: BinaryHeatmapMethod.__build_heatmap_mp(
            kp,
            xy_grid,
            radius,
            destination_call=fn_c
        )

        # Decide whether to perform calucalation in a batch dimension.
        # May be faster, but requires more memory.
        if len(kp.get_shape()) == 4:  # [b, c, h, w]
            return fn_b(kp, xy_grid, radius)
        elif len(kp.get_shape()) == 3:  # [c, h, w]
            return fn_c(kp, xy_grid, radius)
        else:
            message = f'Expected keypoints dimensionality to be 3 or 4, but got {len(kp.get_shape())}.' + \
                      f'Keypoints shape: {kp.get_shape()}'
            raise Exception(message)

    @staticmethod
    def __build_heatmap_mp(kp, xy_grid, radius, destination_call):
        """
        The hetmaps generation is factorized down to single keypoint heatmap generation.
        Nested calls of this method allow for highly optimized vectorized computation of multiple maps.

        Parameters
        ----------
        kp : tf.Tensor of shape [..., 2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        radius : tf.float32
            Radius of the classification (heat) region.
        destination_call : method pointer
            Used for nested calling to increase the dimensionality of the computation.
        """
        fn = lambda _kp: destination_call(_kp, xy_grid, radius)
        maps = tf.vectorized_map(fn, kp)
        return maps

    def __build_heatmap(self, kp, xy_grid, radius):
        """
        Builds a hard classification heatmap for a single keypoint `kp`.
        Parameters
        ----------
        kp : tf.Tensor of shape [2]
            A keypoint (x, y) for which to build the heatmap.
        xy_grid : tf.Tensor of shape [h, w, 2]
            A coordinate grid for the image tensor.
        radius : tf.float32
            Radius of the classification (heat) region.
        """
        print(kp.get_shape())
        grid_size = xy_grid.get_shape()[:2]
        heatmap = tf.ones((grid_size[0], grid_size[1]), dtype=self.map_dtype)

        bool_location_map = (xy_grid[..., 0] - kp[0]) ** 2 + (xy_grid[..., 1] - kp[1]) ** 2 < radius ** 2
        bool_location_map = tf.cast(bool_location_map, dtype=self.map_dtype)
        return heatmap * bool_location_map


class FlipPostMethod(TFRPostMapMethod):
    def __init__(self, keypoints_map, rate=0.5):
        """
        Flip image and its keypoints with the probability of `rate`.
        Parameters
        ----------
        keypoints_map : list
            Contains mappings which describe the transformation of points after the flip.
            Examples: [[0, 1], [2, 4]] means, that point with index 0 become a point with index 1
            and point with index 2 becomes a point with index 4.
            Warning! len(keypoints_map) must be equal to the total number of points.
        rate : float
            The probability of flip.
        """
        super().__init__()
        self._rate = rate
        keypoints_map = sorted(keypoints_map, key=lambda x: x[0])
        keypoints_map = [x[1] for x in keypoints_map]
        self._keypoints_map = keypoints_map

    def flip(self, image, absent_human_mask, keypoints, masks):
        """
        Parameters
        ----------
        keypoints : tf.Tensor of shape [batch, c, n_people, 2]
            Tensor of keypoints coordinates.
        """
        # Flip the image and its corresponding absent human mask
        flipped_im = tf.image.flip_left_right(image)
        flipped_ah_mask = tf.image.flip_left_right(absent_human_mask)
        # Flip keypoints
        _, height, width, _ = image.get_shape().as_list()
        move = np.array([[[width, 0]]], dtype='float32')
        keypoints = move - keypoints
        # Flip y coordinate since it has changed its sign
        scale_y = np.array([[[1, -1]]], dtype='float32')
        keypoints = keypoints * scale_y

        # Reorder points and their masks
        keypoints = tf.gather(keypoints, self._keypoints_map, axis=1)
        masks = tf.gather(masks, self._keypoints_map, axis=1)
        return flipped_im, flipped_ah_mask[..., :1], keypoints, masks

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]
        absent_human_mask = element[RIterator.ABSENT_HUMAN_MASK]
        keypoints = element[RIterator.KEYPOINTS]
        masks = element[RIterator.KEYPOINTS_MASK]

        p = tf.random_uniform(minval=0, maxval=1.0, shape=[])
        true_fn = lambda: self.flip(image, absent_human_mask, keypoints, masks)
        false_fn = lambda: (image, absent_human_mask, keypoints, masks)
        image, absent_human_mask, keypoints, masks = tf.cond(p < self._rate, true_fn, false_fn)

        element[RIterator.ABSENT_HUMAN_MASK] = absent_human_mask
        element[RIterator.IMAGE] = image
        element[RIterator.KEYPOINTS] = keypoints
        element[RIterator.KEYPOINTS_MASK] = masks
        return element


class ImageAdjustPostMethod(TFRPostMapMethod):
    MSG_NORM_IMAGE = "Image is normalized. Cannot change brightness and contrast."

    def __init__(self, contrast_factor_range=(0.5, 2.0), max_delta=0.4, contrast_rate=0.5, brightness_rate=0.5, assert_image=True):
        """
        Does contrast and brightness adjustment.

        Parameters
        ----------
        contrast_factor_range : tuple
        max_delta : tuple
        contrast_rate : float
            Probability of changing contrast.
        brightness_rate : float
            Probability of changing brightness.
        assert_image : bool
            When changing brightness and contrast, it will be checked whether the image is normalized.
            If the image is normalized, an exception is thrown. The check is done via looking at the mean
            of the image: if it greater than 3.0, then the image is unnormalized and everything is okay.
        """
        super().__init__()
        self._cont_low = contrast_factor_range[0]
        if self._cont_low <= 0.0:
            raise ValueError(f'The lowest value for contrast factor must be positive, received {self._cont_low}')

        self._cont_high = contrast_factor_range[1]
        self._max_delta = max_delta
        if self._max_delta <= 0.0:
            raise ValueError(f'The max_delta value for brightness must be positive, received {self._max_delta}')

        self._contrast_rate = contrast_rate
        self._brightness_rate = brightness_rate
        self._assert_image = assert_image

    def adjust_contrast(self, image):
        return tf.image.random_contrast(image, lower=self._cont_low, upper=self._cont_high)

    def adjust_brightness(self, image):
        return tf.image.random_brightness(image, max_delta=self._max_delta,)

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]

        if self._assert_image:
            # Make sure the image is unnormalized
            normalization_check = tf.assert_greater(tf.reduce_mean(image), 3.0, message=ImageAdjustPostMethod.MSG_NORM_IMAGE)
            with tf.control_dependencies([normalization_check]):
                image = self.adjust_image(image)
        else:
            image = self.adjust_image(image)

        element[RIterator.IMAGE] = image
        return element

    def adjust_image(self, image):
        # Random contrast and random brightness work only with integer values
        image = tf.cast(image, tf.uint8)

        p = tf.random.uniform(minval=0, maxval=1, shape=[])
        true_fn = lambda: self.adjust_contrast(image)
        false_fn = lambda: image
        image = tf.cond(p < self._contrast_rate, true_fn, false_fn)

        p = tf.random.uniform(minval=0, maxval=1, shape=[])
        true_fn = lambda: self.adjust_brightness(image)
        false_fn = lambda: image
        image = tf.cond(p < self._brightness_rate, true_fn, false_fn)

        # Cast the image back to float
        image = tf.cast(image, tf.float32)

        return image


class ResizePostMethod(TFRPostMapMethod):
    _EXCEPTION_INTERPOLATION_IS_NOT_FOUND = "Interpolation {} does not exist"

    INTERPOLATION_BILINEAR = 'bilinear'
    INTERPOLATION_NEAREST_NEIGHBOR = 'nearest_neighbor'
    INTERPOLATION_AREA = 'area'
    INTERPOLATION_BICUBIC = 'bicubic'

    FIELD_INTERPOLATION = 'interpolation'

    NAME2METHOD = {
        INTERPOLATION_BILINEAR: tf.image.resize_bilinear,
        INTERPOLATION_NEAREST_NEIGHBOR: tf.image.resize_nearest_neighbor,
        INTERPOLATION_AREA: tf.image.resize_area,
        INTERPOLATION_BICUBIC: tf.image.resize_bicubic,
    }

    def __init__(self, resize_to: list, interpolation='bilinear'):
        """
        Resize image to certain size

        Parameters
        ----------
        resize_to : list
            List [H_new, W_new] - new size for image after resize operation
        interpolation : str
            One of type resize images. ('bilinear', 'nearest_neighbor', 'area', 'bicubic')

        """
        super().__init__()
        self._resize_to = resize_to
        self._interpolation = interpolation

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]
        image_mask = element[RIterator.ABSENT_HUMAN_MASK]
        keypoints = element[RIterator.KEYPOINTS]
        keypoints_mask = element[RIterator.KEYPOINTS_MASK]

        interpolation_method = ResizePostMethod.NAME2METHOD.get(self._interpolation)
        if interpolation_method is not None:
            # Resize image and image mask
            if len(image.get_shape().as_list()) == 4:
                # Batched
                old_hw = tf.shape(image)[1:-1]
                resized_image = interpolation_method(image, self._resize_to)
                resized_mask = interpolation_method(image_mask, self._resize_to)
            else:
                # Single
                old_hw = tf.shape(image)[:-1]
                resized_image = interpolation_method(image, self._resize_to)[0]
                resized_mask = interpolation_method(image_mask, self._resize_to)[0]

            # new / old
            scale_for_keypoints = tf.cast(
                tf.stack(
                    [
                        self._resize_to[1] / old_hw[1],
                        self._resize_to[0] / old_hw[0]
                    ],
                    axis=0
                ),
                dtype=tf.float32
            )

            scaled_keypoints = keypoints * scale_for_keypoints

        else:
            raise Exception(
                ResizePostMethod._EXCEPTION_INTERPOLATION_IS_NOT_FOUND.format(self._interpolation)
            )

        element[RIterator.IMAGE] = resized_image
        element[RIterator.ABSENT_HUMAN_MASK] = resized_mask
        element[RIterator.KEYPOINTS] = scaled_keypoints
        # Check which keypoints are beyond the image
        element[RIterator.KEYPOINTS_MASK] = keypoints_mask * check_bounds(scaled_keypoints, self._resize_to)
        return element


class ApplyMaskToImagePostMethod(TFRPostMapMethod):

    def __init__(self):
        """
        Apply mask to image,
        The main purpose of this is to remove non-labeled people

        """
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]
        image_mask = element[RIterator.ABSENT_HUMAN_MASK]

        element[RIterator.IMAGE] = image * image_mask
        return element


class DropBlockPostMethod(TFRPostMapMethod):

    ZERO = 0
    COLOR_DIMS_DEFAULT = 3

    def __init__(self, min_size_box, max_size_box, prob_rate=0.5, show_kp_under_dropblock=True):
        """
        Cutout (fill it with zeros) random square on the image

        Parameters
        ----------
        min_size_box : tuple or int
            Minimum size of a cutout box, tuple - (Height, Width),
            If only int value provided, then height and width will be equal  to `min_size_box`,
            i.e. it will be square with certain border size
        max_size_box : tuple or int
            Maximum size of a cutout box, tuple - (Height, Width),
            If only int value provided, then height and width will be equal  to `min_size_box`,
            i.e. it will be square with certain border size
        prob_rate : float
            Probability to cutout block of image.
            Must be in range (0, 1)
        show_kp_under_dropblock : bool
            If equal tu True, keypoint under dropblock will be shown to NN as it is

        """
        if prob_rate < 0.0 or prob_rate > 1.0:
            raise ValueError(f"Wrong value for `prob_rate`. Value must be in range (0, 1), but {prob_rate} were given")

        if isinstance(min_size_box, int):
            min_size_box = [min_size_box, min_size_box]
        elif not (isinstance(min_size_box, tuple) or isinstance(min_size_box, list)):
            raise TypeError("Wrong input type for `min_size_box`, it must be int or tuple with size 2\n"
                            f"but were given type: {type(min_size_box)} with value: {min_size_box}")

        if isinstance(max_size_box, int):
            max_size_box = [max_size_box, max_size_box]
        elif not (isinstance(max_size_box, tuple) or isinstance(max_size_box, list)):
            raise TypeError("Wrong input type for `max_size_box`, it must be int or tuple with size 2\n"
                            f"but were given type: {type(max_size_box)} with value: {max_size_box}")

        if min_size_box[0] > max_size_box[0]:
            raise ValueError("Wrong value for `min_size_box` at 0 index (Height). "
                             f"Value must be lower than `max_size_box[0]`, i.e. max Height, "
                             f"but {max_size_box[0]} were given."
            )

        if min_size_box[1] > max_size_box[1]:
            raise ValueError("Wrong value for `min_size_box` at 1 index (Width). "
                             f"Value must be lower than `max_size_box[1]`, i.e. max Width, "
                             f"but {max_size_box[1]} were given."
            )

        self.__min_size_box = min_size_box
        self.__max_size_box = max_size_box
        self.__prob_rate = prob_rate
        self.__show_kp_under_dropblock = show_kp_under_dropblock
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]

        # This method work only with batch of images
        if len(image.get_shape().as_list()) != 4:
            raise TypeError("Input image must be batched, i.e. must have 4 dims!,"
                            f"But {len(image.get_shape().as_list())} were given")
        # For easy access
        box_min = self.__min_size_box
        box_max = self.__max_size_box

        N_batch, height, width = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]

        # Generate size of box
        box_w = tf.random.uniform([N_batch], box_min[1], box_max[1], tf.int32)
        box_h = tf.random.uniform([N_batch], box_min[0], box_max[0], tf.int32)
        # Shape: [N, 2] -> [N, 1, 2] -> [N, 1, 1, 2]
        box_size = tf.expand_dims(tf.expand_dims(tf.stack([box_w, box_h], axis=-1), axis=1), axis=1)
        # Generate left corner of position where should be started cutout op
        h = tf.map_fn(lambda x: tf.random.uniform([], DropBlockPostMethod.ZERO, height - x, tf.int32), box_h)
        w = tf.map_fn(lambda x: tf.random.uniform([], DropBlockPostMethod.ZERO, width  - x, tf.int32), box_w)
        wh_tf = tf.expand_dims(tf.expand_dims(tf.stack([w, h], axis=-1), axis=1), axis=1)
        # Generate grid for cutout
        x_grid, y_grid = tf.meshgrid(tf.range(width), tf.range(height))
        xy_grid = tf.expand_dims(tf.stack([x_grid, y_grid], axis=-1), axis=0)

        # Cutout area that bigger and lower of generated bounds
        # Bigger
        coord_block_b = tf.greater(xy_grid, wh_tf)
        bool_ans_b = tf.math.logical_and(coord_block_b[..., 0], coord_block_b[..., 1])
        # Lower
        coord_block_l = tf.less(xy_grid, wh_tf + box_size)
        bool_ans_l = tf.math.logical_and(coord_block_l[..., 0], coord_block_l[..., 1])
        # Bigger AND Lower
        bool_final = tf.math.logical_and(bool_ans_l, bool_ans_b)

        # Append color shape
        add_color_dims_boolean = tf.concat(
            [tf.expand_dims(bool_final, axis=-1)] * DropBlockPostMethod.COLOR_DIMS_DEFAULT,
            axis=-1
        )
        # Apply mask
        # `where` in this case faster than simple multiplication, because we must cast boolean array
        get_cutout_image = lambda: tf.where(
            add_color_dims_boolean,
            tf.zeros_like(image, dtype=image.dtype),
            image
        )

        p = tf.random.uniform(minval=0, maxval=1, shape=[])
        final_image = tf.cond(
            p < self.__prob_rate,
            get_cutout_image, # True
            lambda: image,    # False
        )

        element[RIterator.IMAGE] = final_image
        if not self.__show_kp_under_dropblock:
            keypoints = element[RIterator.KEYPOINTS]
            keypoints_mask = element[RIterator.KEYPOINTS_MASK]
            element[RIterator.KEYPOINTS_MASK] = keypoints_mask * tf.expand_dims(cutout_kp_in_box(keypoints, wh_tf, box_size), axis=-1)
        return element


class NoisePostMethod(TFRPostMapMethod):

    def __init__(self, std=1.0, mean=0.0, prob_rate=0.5):
        """
        Add noise to image

        Parameters
        ----------
        std : float
            Std for tf.random_normal method
        mean : float
            Mean for tf.random_normal method
        prob_rate : float
            Probability to apply noise to image.
            Must be in range (0, 1)

        """
        if prob_rate < 0.0 or prob_rate > 1.0:
            raise ValueError(f"Wrong value for `prob_rate`. Value must be in range (0, 1), "
                             f"but {prob_rate} were given"
            )

        self.__std = std
        self.__mean = mean
        self.__prob_rate = prob_rate
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        if self._parent_method is not None:
            element = self._parent_method.read_record(serialized_example)
        else:
            element = serialized_example
        image = element[RIterator.IMAGE]

        noise = tf.random_normal(shape=tf.shape(image), mean=self.__mean, stddev=self.__std, dtype=tf.float32)
        apply_image_noised = lambda: tf.add(tf.cast(image, dtype=tf.float32), noise)

        p = tf.random.uniform(minval=0, maxval=1, shape=[])
        final_image = tf.cond(
            p < self.__prob_rate,
            apply_image_noised, # True
            lambda: image,      # False
        )

        element[RIterator.IMAGE] = final_image
        return element


class BackgroundAugmentMethod(TFRPostMapMethod):

    MAX_VALUE_IMAGE = 255

    def __init__(self, backpool_path: str, crop_h: int, crop_w: int):
        """
        TODO: add docs

        Parameters
        ----------
        backpool_path : str
            Path for pool of backgrounds
        crop_h : int
            Height of the crop.
        crop_w : int
            Width of the crop.

        """
        super().__init__()
        self._image_path_pool = glob.glob(os.path.join(backpool_path, '*'))
        self._image_path_tf_constants_pool = tf.constant(
            self._image_path_pool,
            dtype=tf.string
        )
        self._image_crop_size = [crop_h, crop_w, 3]
        self._image_crop_size_tf = tf.constant(np.array([crop_h, crop_w, 3], dtype=np.int32))

    def pick_background(self) -> tf.Tensor:
        random_index = tf.random_uniform(
            shape=[],
            dtype=tf.int32,
            # it is unlikely that a tensor with shape more than 10000 will appear
            maxval=10000
        ) % len(self._image_path_pool)
        background_decoded = tf.io.read_file(self._image_path_tf_constants_pool[random_index])
        background = tf.image.decode_image(background_decoded)
        # crop and return
        # This is an adapted code from the original TensorFlow's `random_crop` method
        limit = tf.shape(background) - self._image_crop_size_tf + 1
        offset = tf.random_uniform(
            shape=[3],
            dtype=tf.int32,
            # it is unlikely that a tensor with shape more than 10000 will appear
            maxval=10000
        ) % limit

        cropped_background = tf.slice(background, offset, self._image_crop_size_tf)

        # After slicing the tensors doesn't have proper shape. They get instead [None, None, None].
        # We can't use tf.Tensors for setting shape because they are note iterable what causes errors.
        cropped_background.set_shape(self._image_crop_size)
        return cropped_background

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        image = tf.cast(element[RIterator.IMAGE], dtype=tf.int32)
        # Smash image into binary (0 and 1) values!
        alpha_image = tf.cast(element[RIterator.ALPHA_MASK], dtype=tf.int32)
        # Smash image into binary (0 and 1) values!
        alpha_image = alpha_image // BackgroundAugmentMethod.MAX_VALUE_IMAGE

        background = tf.cast(self.pick_background(), dtype=tf.int32)
        new_image = image * alpha_image + background * (1 - alpha_image)

        element[RIterator.IMAGE] = tf.cast(new_image, dtype=tf.float32)
        return element
