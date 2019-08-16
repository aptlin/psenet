import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import psenet.config as config
import psenet.preprocess as preprocess


class Dataset:
    def __init__(
        self,
        dataset_dir,
        batch_size,
        resize_length=config.RESIZE_LENGTH,
        crop_size=config.CROP_SIZE,
        min_scale=config.MIN_SCALE,
        kernel_num=config.KERNEL_NUM,
        num_readers=config.NUM_READERS,
        should_shuffle=False,
        should_repeat=False,
        should_augment=True,
        input_context=None,
        prefetch=config.PREFETCH,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.resize_length = resize_length
        self.num_readers = num_readers
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat
        self.min_scale = min_scale
        self.kernel_num = kernel_num
        self.crop_size = crop_size
        self.should_augment = should_augment
        self.input_context = input_context
        self.prefetch = prefetch

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
            "image/text/tags/encoded": tf.io.FixedLenFeature(
                (), tf.string, default_value=""
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
        n_bboxes = tf.cast(parsed_features["image/text/boxes/count"], "int64")
        bboxes_shape = tf.stack([n_bboxes, config.BBOX_SIZE])
        bboxes = tf.reshape(bboxes, bboxes_shape)
        image_name = parsed_features["image/filename"]
        if image_name is None:
            image_name = tf.constant("")
        tags = parsed_features["image/text/tags/encoded"]
        sample = {
            config.BBOXES: bboxes,
            config.HEIGHT: parsed_features["image/height"],
            config.IMAGE_NAME: image_name,
            config.IMAGE: image,
            config.WIDTH: parsed_features["image/width"],
            config.TAGS: tags,
        }
        return sample

    def _process_tagged_bboxes(self, bboxes, tags, height, width):
        tags = str(tags)
        gt_text = np.zeros([height, width], dtype="uint8")
        mask = np.ones([height, width], dtype="uint8")
        bboxes_count, num_points = np.asarray(bboxes.shape).astype("int64")[:2]
        if bboxes_count > 0:
            bboxes = np.reshape(
                bboxes * ([width, height] * 4),
                (bboxes_count, int(num_points / 2), 2),
            ).astype("int32")
            for i in range(bboxes_count):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if tags[i] == "0":
                    cv2.drawContours(mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros([height, width], dtype="uint8")
            kernel_bboxes = preprocess.shrink(bboxes, rate)
            for i in range(bboxes_count):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)
        return gt_kernels, gt_text, mask

    def _preprocess_example(self, sample):
        image = sample[config.IMAGE]
        tags = sample[config.TAGS]
        bboxes = sample[config.BBOXES]

        if self.should_augment:
            image = preprocess.random_scale(
                image,
                resize_length=self.resize_length,
                crop_size=self.crop_size,
            )
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        processed = tf.py_function(
            func=self._process_tagged_bboxes,
            inp=[bboxes, tags, height, width],
            Tout=[tf.uint8, tf.uint8, tf.uint8],
        )
        gt_kernels = processed[0]
        gt_text = processed[1]
        mask = processed[2]

        if self.should_augment:
            tensors = [image, gt_text, mask]
            for idx in range(1, self.kernel_num):
                tensors.append(gt_kernels[idx - 1])
            tensors = preprocess.random_flip(tensors)
            tensors = preprocess.random_rotate(tensors)
            tensors = preprocess.random_background_crop(
                tensors, crop_size=self.crop_size
            )
            image, gt_text, mask, gt_kernels = (
                tensors[0],
                tensors[1],
                tensors[2],
                tensors[3:],
            )
            image = tf.image.random_brightness(image, 32 / 255)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        gt_text = tf.cast(gt_text, tf.float32)
        gt_text = tf.sign(gt_text)
        gt_text = tf.cast(gt_text, tf.uint8)
        gt_text = tf.expand_dims(gt_text, axis=0)
        image = tf.cast(image, tf.float32)
        label = tf.concat([gt_text, gt_kernels], axis=0)
        label = tf.transpose(label, perm=[1, 2, 0])
        label = tf.cast(label, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return ({config.IMAGE: image, config.MASK: mask}, label)

    def _guarantee_validity(self, inputs, label):
        def is_valid(side):
            return tf.logical_and(
                tf.math.greater_equal(side, config.MIN_SIDE),
                tf.math.equal(tf.math.floormod(side, config.MIN_SIDE), 0),
            )

        image = inputs[config.IMAGE]
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        return tf.logical_and(is_valid(height), is_valid(width))

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
        dataset = dataset.map(
            self._preprocess_example, num_parallel_calls=self.num_readers
        )

        dataset = dataset.filter(self._guarantee_validity)

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                {config.IMAGE: [None, None, 3], config.MASK: [None, None]},
                [None, None, config.KERNEL_NUM],
            ),
        ).prefetch(self.prefetch)

        return dataset


def build_dataset(mode, FLAGS):
    def input_fn(input_context=None):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if FLAGS.augment_training_data:
            should_augment = is_training
        else:
            should_augment = False
        dataset_dir = (
            FLAGS.training_data_dir if is_training else FLAGS.eval_data_dir
        )
        dataset = Dataset(
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
