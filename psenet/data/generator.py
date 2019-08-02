import cv2
import numpy as np
import os
import tensorflow as tf
from psenet import config
from psenet.data.utils import preprocess

tf.compat.v1.enable_eager_execution()


class Dataset:
    def __init__(
        self,
        dataset_dir,
        batch_size,
        max_side_length=config.RESIZE_LENGTH,
        min_scale=config.MIN_SCALE,
        kernel_num=config.KERNEL_NUM,
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
        self.min_scale = min_scale
        self.kernel_num = kernel_num

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

    def _process_tagged_bboxes(self, image, bboxes, tags):
        image = preprocess.random_scale(image)
        image = preprocess.scale(image)
        height, width = np.asarray(image.shape).astype("int64")[:2]
        gt_text = np.zeros([height, width], dtype="uint8")
        training_mask = np.ones([height, width], dtype="uint8")
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(
                bboxes * ([width, height] * 4),
                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2),
            ).astype("int32")
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if tags[i] != "1":
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros([height, width], dtype="uint8")
            kernel_bboxes = preprocess.shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)
        tensors = [image, gt_text, training_mask]
        tensors.extend(gt_kernels)
        tensors = preprocess.random_flip(tensors)
        tensors = preprocess.random_rotate(tensors)
        tensors = preprocess.background_random_crop(tensors, gt_text)
        image, gt_text, training_mask, gt_kernels = (
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3:],
        )
        gt_text = np.asarray(gt_text)
        gt_text[gt_text > 0] = 1
        image = tf.image.random_brightness(image, 32 / 255)
        image = tf.image.random_saturation(image, 0.5, 1.5)
        image = preprocess.normalize(image)

        return image, gt_kernels, gt_text, training_mask

    def _preprocess_example(self, sample):
        image = sample[config.IMAGE]
        tags = sample[config.TAGS]
        bboxes = sample[config.BBOXES]
        processed = tf.numpy_function(
            func=self._process_tagged_bboxes,
            inp=[image, bboxes, tags],
            Tout=[tf.float32, tf.uint8, tf.uint8, tf.uint8],
        )
        image = processed[0]
        gt_kernels = processed[1]
        gt_text = processed[2]
        training_mask = processed[3]
        sample.pop(config.TAGS, None)
        sample.pop(config.BBOXES, None)
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        sample[config.HEIGHT] = height
        sample[config.WIDTH] = width
        sample[config.IMAGE] = image
        sample[config.KERNELS] = gt_kernels
        sample[config.TEXT] = gt_text
        sample[config.TRAINING_MASK] = training_mask
        return sample

    def _get_all_tfrecords(self):
        return tf.io.gfile.glob(os.path.join(self.dataset_dir, "*.tfrecord"))

    def get_one_shot_iterator(self):
        tfrecords = self._get_all_tfrecords()
        dataset = (
            tf.data.TFRecordDataset(
                tfrecords, num_parallel_reads=self.num_readers
            )
            .map(self._parse_example, num_parallel_calls=self.num_readers)
            .map(self._preprocess_example, num_parallel_calls=self.num_readers)
        )

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        # dataset = dataset.padded_batch(
        #     self.batch_size,
        #     padded_shapes={
        #         config.HEIGHT: [],
        #         config.IMAGE_NAME: [],
        #         config.IMAGE: [None, None, 3],
        #         config.TRAINING_MASK: [None, None],
        #         config.TEXT: [None, None],
        #         config.KERNELS: [None, None, config.KERNEL_NUM - 1],
        #         config.WIDTH: [],
        #     },
        # ).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()
