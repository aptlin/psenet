import os
import tensorflow as tf
from psenet import config


# Default file pattern of TFRecord of TensorFlow Example.
class Dataset:
    def __init__(
        self,
        dataset_dir,
        batch_size,
        max_side_length=config.RESIZE_LENGTH,
        num_readers=1,
        should_shuffle=False,
        should_repeat=False,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.max_side_length = max_side_length
        self.num_readers = num_readers
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat

    def _parse_example(self, example_prototype):
        features = {
            "image/encoded": tf.io.FixedLenFeature(
                (), tf.string, default_value=""
            ),
            "image/filename": tf.io.FixedLenFeature(
                (), tf.string, default_value=""
            ),
            "image/format": tf.io.FixedLenFeature(
                (), tf.string, default_value="jpeg"
            ),
            "image/height": tf.io.FixedLenFeature(
                (), tf.int64, default_value=0
            ),
            "image/width": tf.io.FixedLenFeature(
                (), tf.int64, default_value=0
            ),
            "image/text/boxes/count": tf.io.FixedLenFeature(
                (), tf.int64, default_value=0
            ),
            "image/text/boxes/encoded": tf.io.VarLenFeature(tf.float32),
        }
        parsed_features = tf.io.parse_single_example(
            example_prototype, features
        )
        image_data = parsed_features["image/encoded"]
        image = tf.cond(
            tf.image.is_jpeg(image_data),
            lambda: tf.image.decode_jpeg(image_data, 3),
            lambda: tf.image.decode_png(image_data, 3),
        )
        bboxes = tf.sparse.to_dense(
            parsed_features["image/text/boxes/encoded"], default_value=0
        )
        n_bboxes = tf.cast(
            parsed_features["image/text/boxes/count"] / config.BBOX_SIZE,
            "int32",
        )
        bboxes_shape = tf.stack([n_bboxes, config.BBOX_SIZE])
        bboxes = tf.reshape(bboxes, bboxes_shape)
        image_name = parsed_features["image/filename"]
        if image_name is None:
            image_name = tf.constant("")
        sample = {
            config.BBOXES: bboxes,
            config.HEIGHT: parsed_features["image/height"],
            config.IMAGE_NAME: image_name,
            config.IMAGE: image,
            config.NUMBER_OF_BBOXES: n_bboxes,
            config.WIDTH: parsed_features["image/width"],
        }
        return sample

    def _preprocess_example(self, sample):
        image = sample[config.IMAGE]
        sample[config.IMAGE] = image
        return sample

    def _get_all_tfrecords(self):
        return tf.io.gfile.glob(os.path.join(self.dataset_dir, "*.tfrecord"))

    def get_one_shot_iterator(self):
        tfrecords = self._get_all_tfrecords()
        dataset = tf.data.TFRecordDataset(
            tfrecords, num_parallel_reads=self.num_readers
        ).map(self._parse_example, num_parallel_calls=self.num_readers)

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes={
                config.BBOXES: [None, 8],
                config.HEIGHT: [],
                config.IMAGE_NAME: [],
                config.IMAGE: [None, None, 3],
                config.NUMBER_OF_BBOXES: [],
                config.WIDTH: [],
            },
        ).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()
