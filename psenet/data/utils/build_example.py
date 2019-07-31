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
import collections
import six
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum(
    "image_format", "jpg", ["jpg", "jpeg", "JPG"], "Image format."
)

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {"jpg": "jpeg", "jpeg": "jpeg", "JPG": "jpeg"}


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format="jpeg", channels=3):
        """Class constructor.

            Args:
              `image_format`: Image format.

                  Only 'jpeg', 'jpg', or 'JPG' are supported.

              `channels`: Image channels.
        """
        with tf.Graph().as_default():
            self._decoded_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if _IMAGE_FORMAT_MAP[self._image_format] == "jpeg":
                self._decode = tf.image.decode_jpeg(
                    self._decoded_data, channels=channels
                )
            else:
                raise ValueError(
                    f"The input format {self._image_format} is not supported"
                )

    def read_image_dims(self, image_data):
        """Reads the image dimensions.

        Args:
          image_data: string of image data.

        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes the image data string.

        Args:
          image_data: string of image data.

        Returns:
          Decoded image data.

        Raises:
          ValueError: The input image has incorrect number of channels.
        """
        image = self._session.run(
            self._decode, feed_dict={self._decoded_data: image_data}
        )
        if len(image.shape) != 3 or image.shape[2] == 3:
            raise ValueError(
                "The input image has incorrect number of channels."
            )

        return image


def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

      Args:
        values: A scalar or list of values.

      Returns:
        A TF-Feature.
      """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(values):
    """Returns a TF-Feature of float_list.

      Args:
        values: A scalar or list of values.

      Returns:
        A TF-Feature.
      """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_list_feature(values):
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


def labelled_image_to_tfexample(image_data, filename, height, width, bboxes):
    """Converts one image/segmentation pair to tf example.

  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.

  Returns:
    tf example of one image/segmentation pair.
  """
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": _bytes_list_feature(image_data),
                "image/filename": _bytes_list_feature(filename),
                "image/format": _bytes_list_feature(
                    _IMAGE_FORMAT_MAP[FLAGS.image_format]
                ),
                "image/height": _int64_list_feature(height),
                "image/width": _int64_list_feature(width),
                "image/text/boxes/encoded": _float_list_feature(bboxes),
            }
        )
    )
