from ..pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from .data_preparation import IMAGE_FNAME, KEYPOINTS_FNAME, KEYPOINTS_MASK_FNAME, IMAGE_PROPERTIES_FNAME
from .utils import check_bounds

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
        """
        self.shape_keypoints = shape_keypoints
        self.shape_image_properties = shape_image_properties
        self.shape_keypoints_mask = shape_keypoints[:-1] + [1]

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

    def __init__(self, crop_w: int, crop_h: int):
        """
        Perform random crop of the input images and their corresponding uvmaps.
        Parameters
        ----------
        crop_w : int
            Width of the crop.
        crop_h : int
            Height of the crop.
        """
        super().__init__()
        self._crop_w = crop_w
        self._crop_h = crop_h
        self._image_crop_size = [crop_h, crop_w, 3]
        self._image_crop_size_tf = tf.constant(np.array([crop_h, crop_w, 3], dtype=np.int32))

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

        element[RIterator.IMAGE] = cropped_image
        element[RIterator.KEYPOINTS] = cropped_keypoints
        element[RIterator.KEYPOINTS_MASK] = keypoints_mask * check_bounds(cropped_keypoints, self._image_crop_size_tf)
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

