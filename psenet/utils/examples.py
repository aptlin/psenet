"""Contains common utility functions and classes for building the dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/text/boxes/encoded: encoded bounding boxes.
"""
from collections.abc import Iterable

import six
import tensorflow as tf


def int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

      Args:
        values: A scalar or list of values.

      Returns:
        A TF-Feature.
      """
    if not isinstance(values, Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_list_feature(values):
    """Returns a TF-Feature of float_list.

      Args:
        values: A scalar or list of values.

      Returns:
        A TF-Feature.
      """
    if not isinstance(values, Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_list_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """

    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)])
    )
