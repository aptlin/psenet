import tensorflow as tf


class ImageReader:
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self,):
        self._image_format_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}
        with tf.Graph().as_default():
            self._decode_data = tf.compat.v1.placeholder(dtype=tf.string)
            self._session = tf.compat.v1.Session()

    def _decode_image_data(self, image_data, image_format, channels=3):
        image_format = image_format.lower()
        if self._image_format_map[image_format] == "jpeg":
            decoder = tf.image.decode_jpeg(
                self._decode_data, channels=channels
            )
        elif self._image_format_map[image_format] == "png":
            decoder = tf.image.decode_png(self._decode_data, channels=channels)
        else:
            raise ValueError(
                f"The input format {self._image_format} is not supported"
            )
        return self._session.run(
            decoder, feed_dict={self._decode_data: image_data}
        )

    def decode_image(self, image_data, image_format, channels=3):
        """Decodes the image data string.

        Args:
          image_data: string of image data.

        Returns:
          Decoded image data.

        Raises:
          ValueError: The input image has incorrect number of channels.
        """
        image = self._decode_image_data(image_data, image_format, channels)
        if len(image.shape) != 3:
            raise ValueError(
                "The input image shape is incorrect, "
                + "got {} dimensions, expected 3.".format(len(image.shape))
            )
        if image.shape[2] != channels:
            raise ValueError(
                "The number of channels is incorrect, "
                + "got {}, expected {}.".format(image.shape[2], channels)
            )
        return image

    def read_image_dims(self, image_data, image_format, channels=3):
        """Reads the image dimensions.

        Args:
          image_data: string of image data.

        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data, image_format, channels)
        return image.shape[:2]

    def run(self, image_derivative):
        return self._session.run(image_derivative)
