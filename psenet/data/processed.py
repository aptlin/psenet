import os

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

import psenet.config as config


class ProcessedDataset:
    def __init__(
        self,
        dataset_dir,
        batch_size,
        kernel_num=config.KERNEL_NUM,
        num_readers=config.NUM_READERS,
        should_shuffle=False,
        should_repeat=False,
        input_context=None,
        prefetch=config.PREFETCH,
    ):
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.input_context = input_context
        self.kernel_num = kernel_num
        self.num_readers = num_readers
        self.prefetch = prefetch
        self.should_repeat = should_repeat
        self.should_shuffle = should_shuffle

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


def build(mode, FLAGS):
    def input_fn(input_context=None):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if FLAGS.augment_training_data:
            should_augment = is_training
        else:
            should_augment = False
        dataset_dir = (
            FLAGS.training_data_dir if is_training else FLAGS.eval_data_dir
        )
        dataset = ProcessedDataset(
            dataset_dir,
            FLAGS.batch_size,
            resize_length=FLAGS.resize_length,
            min_scale=FLAGS.min_scale,
            kernel_num=FLAGS.kernel_num,
            crop_size=FLAGS.resize_length // 2,
            num_readers=FLAGS.readers_num,
            should_shuffle=is_training,
            should_repeat=True,
            should_augment=should_augment,
            input_context=input_context,
            prefetch=FLAGS.prefetch,
        ).build()
        return dataset

    return input_fn
