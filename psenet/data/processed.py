import os

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from psenet import config


class ProcessedDataset:
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.dataset_dir = FLAGS.dataset_dir
        self.input_context = FLAGS.input_context
        self.kernel_num = FLAGS.kernel_num
        self.num_readers = FLAGS.num_readers
        self.prefetch = FLAGS.prefetch
        self.should_repeat = FLAGS.should_repeat
        self.should_shuffle = FLAGS.should_shuffle

    def _parse_example(self, example_proto):
        features = {
            "height": tf.io.FixedLenFeature((), tf.int64, default_value=0),
            "width": tf.io.FixedLenFeature((), tf.int64, default_value=0),
            "features/image": tf.io.VarLenFeature(tf.float32),
            "features/mask": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        height = parsed_features["height"]
        width = parsed_features["width"]

        image = tf.sparse.to_dense(
            parsed_features["features/image"], default_value=0
        )
        image = tf.reshape(image, [height, width, 3])

        mask = tf.sparse.to_dense(
            parsed_features["features/mask"], default_value=0
        )
        mask = tf.reshape(mask, [height, width])

        labels = tf.sparse.to_dense(parsed_features["labels"], default_value=0)
        labels = tf.reshape(labels, [height, width, self.kernel_num])

        return ({config.IMAGE: image, config.MASK: mask}, labels)

    def _get_all_tfrecords(self):
        return tf.data.Dataset.list_files(
            os.path.join(self.dataset_dir, "*.tfrecord")
        )

    def build(self):
        dataset = self._get_all_tfrecords()
        if self.input_context:
            dataset = dataset.shard(
                self.input_context.num_input_pipelines,
                self.input_context.input_pipeline_id,
            )
            logging.info(
                "Sharding the dataset for the pipeline {} out of {}".format(
                    self.input_context.input_pipeline_id,
                    self.input_context.num_input_pipelines,
                )
            )
        else:
            logging.info("Received no input context.")

        if self.should_repeat:
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(1)

        if self.should_shuffle:
            dataset = dataset.shuffle(
                buffer_size=config.NUM_BATCHES_TO_SHUFFLE * self.batch_size + 1
            )

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=self.num_readers,
            num_parallel_calls=self.num_readers,
        )

        dataset = dataset.map(
            self._parse_example, num_parallel_calls=self.num_readers
        )

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                {config.IMAGE: [None, None, 3], config.MASK: [None, None]},
                [None, None, self.kernel_num],
            ),
        ).prefetch(self.prefetch)

        return dataset


def build(FLAGS):
    def input_fn(input_context=None):
        is_training = FLAGS.mode == tf.estimator.ModeKeys.TRAIN
        if FLAGS.augment_training_data:
            should_augment = is_training
        else:
            should_augment = False
        FLAGS.should_augment = should_augment
        dataset_dir = (
            FLAGS.training_data_dir if is_training else FLAGS.eval_data_dir
        )
        FLAGS.dataset_dir = dataset_dir
        FLAGS.should_repeat = True
        FLAGS.should_shuffle = is_training
        FLAGS.input_context = input_context
        dataset = ProcessedDataset(FLAGS).build()
        return dataset

    return input_fn
