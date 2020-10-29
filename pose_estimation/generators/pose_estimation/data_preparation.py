
from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.pipeline.tfr.utils import _tensor_to_byte_feature


# Save form
SAVE_FORM = "{0}_{1}.tfrecord"


# Feature names
IMAGE_FNAME = 'IMAGE_FNAME'
ABSENT_HUMAN_MASK_FNAME = 'ABSENT_HUMAN_MASK_FNAME'
KEYPOINTS_FNAME = 'KEYPOINTS_FNAME'
KEYPOINTS_MASK_FNAME = 'KEYPOINTS_MASK_FNAME'
IMAGE_PROPERTIES_FNAME = 'IMAGE_PROPERTIES_FNAME'


# Serialize into data point
def serialize_pose_estimation_data_point(
        image_tensor,
        image_mask_tensor,
        keypoints_tensor, 
        keypoints_mask_tensor, 
        image_properties_tensor, 
        sess=None
):
    feature = {
        IMAGE_FNAME: _tensor_to_byte_feature(image_tensor, sess),
        ABSENT_HUMAN_MASK_FNAME: _tensor_to_byte_feature(image_mask_tensor, sess),
        KEYPOINTS_FNAME: _tensor_to_byte_feature(keypoints_tensor, sess),
        KEYPOINTS_MASK_FNAME: _tensor_to_byte_feature(keypoints_mask_tensor, sess),
        IMAGE_PROPERTIES_FNAME: _tensor_to_byte_feature(image_properties_tensor, sess)
    }

    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_pose_estimation_train_data(
        image_tensors,
        image_mask_tensors,
        keypoints_tensors,
        keypoints_mask_tensors,
        image_properties_tensors,
        tfrecord_path, 
        sess=None
):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, (image_tensor, image_mask_tensor, keypoints_tensor,
                keypoints_mask_tensor, image_properties_tensor) in enumerate(
            zip(
            image_tensors,
            image_mask_tensors,
            keypoints_tensors,
            keypoints_mask_tensors,
            image_properties_tensors
            )
        ):
            serialized_data_point = serialize_pose_estimation_data_point(
                image_tensor=image_tensor,
                image_mask_tensor=image_mask_tensor,
                keypoints_tensor=keypoints_tensor,
                keypoints_mask_tensor=keypoints_mask_tensor,
                image_properties_tensor=image_properties_tensor,        
                sess=sess
            )
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_pose_estimation_train_data(
    image_tensors, image_masks, keypoints_tensors, keypoints_mask_tensors, image_properties_tensors,
    prefix, dp_per_record,  sess=None):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points

    Parameters
    ----------
    """
    for i in range(len(image_tensors) // dp_per_record):
        image_tensor_batched = image_tensors[dp_per_record * i: (i + 1) * dp_per_record]
        image_mask_tensor_batched = image_masks[dp_per_record * i: (i + 1) * dp_per_record]
        keypoints_tensor_batched = keypoints_tensors[dp_per_record * i: (i + 1) * dp_per_record]
        keypoints_mask_tensor_batched = keypoints_mask_tensors[dp_per_record * i: (i + 1) * dp_per_record]
        image_properties_tensor_batched = image_properties_tensors[dp_per_record * i: (i + 1) * dp_per_record]

        tfrecord_name = SAVE_FORM.format(prefix, i)

        record_pose_estimation_train_data(
            image_tensors=image_tensor_batched,
            image_mask_tensor=image_mask_tensor_batched,
            keypoints_tensors=keypoints_tensor_batched,
            keypoints_mask_tensors=keypoints_mask_tensor_batched,
            image_properties_tensors=image_properties_tensor_batched,
            tfrecord_path=tfrecord_name,
            sess=sess
        )

