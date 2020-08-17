from ..pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from .data_preparation import IMAGE_FNAME, KEYPOINTS_FNAME, KEYPOINTS_MASK_FNAME, IMAGE_PROPERTIES_FNAME
from .utils import check_bounds, apply_transformation, apply_transformation_batched

import tensorflow as tf
import numpy as np


class RIterator:
    IMAGE = 'IMAGE'
    KEYPOINTS = 'KEYPOINTS'
    KEYPOINTS_MASK = 'KEYPOINTS_MASK'
    IMAGE_PROPERTIES = 'IMAGE_PROPERTIES'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            shape_keypoints,
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
            Shape of keypoints in tfrecords [n_people, number_of_keypoints]
            For example: [10, 17]
        shape_image_properties : list
            Shape of image properties in tfrecords,
            By default equal to [3]
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
            KEYPOINTS_FNAME: tf.io.FixedLenFeature((), tf.string),
            KEYPOINTS_MASK_FNAME: tf.io.FixedLenFeature((), tf.string),
            IMAGE_PROPERTIES_FNAME: tf.io.FixedLenFeature((), tf.string)
        }

        example = tf.io.parse_single_example(serialized_example, r_feature_description)

        # Extract the data from the example
        image_tensor = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        keypoints_tensor = tf.io.parse_tensor(example[KEYPOINTS_FNAME], out_type=self.keypoints_dtype)

        keypoints_mask_tensor = tf.io.parse_tensor(example[KEYPOINTS_MASK_FNAME], out_type=self.keypoints_mask_dtype)
        image_properties_tensor = tf.io.parse_tensor(example[IMAGE_PROPERTIES_FNAME], out_type=self.image_properties_dtype)

        # Give the data its shape because it doesn't have it right after being extracted
        keypoints_tensor.set_shape(self.shape_keypoints)
        keypoints_mask_tensor.set_shape(self.shape_keypoints_mask)
        image_properties_tensor.set_shape(self.shape_image_properties)

        output_dict = {
            RIterator.IMAGE: image_tensor,
            RIterator.KEYPOINTS: keypoints_tensor,
            RIterator.KEYPOINTS_MASK: keypoints_mask_tensor,
            RIterator.IMAGE_PROPERTIES: image_properties_tensor
        }

        return output_dict


class RandomCropMethod(TFRPostMapMethod):

    def __init__(self, crop_w: int, crop_h: int, image_last_dimension=3):
        """
        Perform random crop of the input images and their corresponding uvmaps.
        Parameters
        ----------
        crop_w : int
            Width of the crop.
        crop_h : int
            Height of the crop.
        image_last_dimension : int
            Number of channels of images, by default equal to 3
        """
        super().__init__()
        self._crop_w = crop_w
        self._crop_h = crop_h
        self._image_crop_size = [crop_h, crop_w, image_last_dimension]
        self._image_crop_size_tf = tf.constant(np.array([crop_h, crop_w, image_last_dimension], dtype=np.int32))

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        image = element[RIterator.IMAGE]
        keypoints = element[RIterator.KEYPOINTS]
        keypoints_mask = element[RIterator.KEYPOINTS_MASK]

        # This is an adapted code from the original TensorFlow's `random_crop` method
        limit = tf.shape(image) - self._image_crop_size_tf + 1
        offset = tf.random_uniform(
            shape=[3],
            dtype=tf.int32,
            # it is unlikely that a tensor with shape more that 10000 will appear
            maxval=10000
        ) % limit

        cropped_image = tf.slice(image, offset, self._image_crop_size_tf)
        cropped_keypoints = keypoints - tf.cast(tf.stack([offset[1], offset[0]]), dtype=tf.float32)
        # After slicing the tensors doesn't have proper shape. They get instead [None, None, None].
        # We can't use tf.Tensors for setting shape because they are note iterable what causes errors.
        cropped_image.set_shape(self._image_crop_size)

        # Check which keypoints are beyond the image
        correct_keypoints_mask = keypoints_mask * check_bounds(cropped_keypoints, self._image_crop_size_tf)

        element[RIterator.IMAGE] = cropped_image
        element[RIterator.KEYPOINTS] = cropped_keypoints
        element[RIterator.KEYPOINTS_MASK] = correct_keypoints_mask
        return element


class AugmentationPostMethod(TFRPostMapMethod):
    
    def __init__(self,
        use_rotation=False,
        angle_min=None,
        angle_max=None,
        use_shift=False,
        dx_min=None,
        dx_max=None,
        dy_min=None,
        dy_max=None,
        use_zoom=False,
        zoom_min=None,
        zoom_max=None
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
        image = element[RIterator.IMAGE]
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

            transformed_image, transformed_keypoints = apply_transformation(
                image,
                keypoints,
                use_rotation=self.use_rotation,
                angle=angle,
                use_shift=self.use_shift,
                dx=dx,
                dy=dy,
                use_zoom=self.use_zoom,
                zoom_scale=zoom
            )
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

            transformed_image, transformed_keypoints = apply_transformation_batched(
                image,
                keypoints,
                use_rotation=self.use_rotation,
                angle_batched=angle,
                use_shift=self.use_shift,
                dx_batched=dx,
                dy_batched=dy,
                use_zoom=self.use_zoom,
                zoom_scale_batched=zoom
            )
            # Check which keypoints are beyond the image
            correct_keypoints_mask = keypoints_mask * check_bounds(transformed_keypoints, image_shape[1:])

        element[RIterator.IMAGE] = transformed_image
        element[RIterator.KEYPOINTS] = transformed_keypoints
        element[RIterator.KEYPOINTS_MASK] = correct_keypoints_mask
        return element


class NormalizePostMethod(TFRPostMapMethod):

    NORMALIZE_KEYPOINTS = 'normalize_keypoints_tensor'
    NORMALIZE_IMAGE = 'normalize_image_tensor'

    def __init__(self, divider=127.5,
                 use_caffee_norm=True,
                 use_float64=True,
                 using_for_image_tensor=False,
                 using_for_image_tensor_only=False):
        """
        Normalizes the tensor by dividing it by the `divider`.
        Parameters
        ----------
        divider : float or int
            The number to divide the tensor by.
        use_float64 : bool
            Set to True if you want the tensor to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        using_for_image_tensor : bool
            If true, divider will be used on tensors for generator.
        """
        super().__init__()
        self.use_float64 = use_float64
        self.use_caffe_norm = use_caffee_norm
        self.using_for_image_tensor = using_for_image_tensor
        self.using_for_image_tensor_only = using_for_image_tensor_only
        if use_float64:
            self.divider = tf.constant(divider, dtype=tf.float64)
        else:
            self.divider = tf.constant(divider, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        target = element[RIterator.KEYPOINTS]
        if not self.using_for_image_tensor_only:
            if self.use_float64:
                target = tf.cast(target, dtype=tf.float64)
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_KEYPOINTS)
                target = tf.cast(target, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_KEYPOINTS)
            element[RIterator.KEYPOINTS] = target

        if self.using_for_image_tensor:
            image_tensor = element[RIterator.IMAGE]
            if self.use_float64:
                image_tensor = tf.cast(image_tensor, dtype=tf.float64)
                if self.use_caffe_norm:
                    image_tensor = (image_tensor - self.divider) / self.divider
                else:
                    image_tensor = tf.divide(image_tensor, self.divider, name=NormalizePostMethod.NORMALIZE_IMAGE)
                image_tensor = tf.cast(image_tensor, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    image_tensor = (image_tensor - self.divider) / self.divider
                else:
                    image_tensor = tf.divide(image_tensor, self.divider, name=NormalizePostMethod.NORMALIZE_IMAGE)
            element[RIterator.IMAGE] = image_tensor

        return element


class RGB2BGRPostMethod(TFRPostMapMethod):

    RGB2BGR_KEYPOINTS = 'RGB2BGR_tensor'
    RGB2BGR_IMAGE = 'BGR2RGB_input'

    def __init__(self, using_for_image_tensor=False):
        """
        Used for swapping color channels in tensors from RGB to BGR format.
        Parameters
        ----------
        using_for_image_tensor : bool
            If true, swapping color channels will be used on input tensors.
        """
        self.using_for_image_tensor = using_for_image_tensor
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        # for tensor
        target = element[RIterator.KEYPOINTS]
        # Swap channels
        element[RIterator.KEYPOINTS] = tf.reverse(target, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_KEYPOINTS)
        # for generator
        if self.using_for_image_tensor:
            image_tensor = element[RIterator.IMAGE]
            # Swap channels
            element[RIterator.IMAGE] = tf.reverse(image_tensor, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_IMAGE)
        return element

