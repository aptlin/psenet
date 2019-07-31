"""Converts the ICDAR MLT 2019 data to TFRecords."""

import math
import os
import random
import sys
import build_example
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "images_dir", "./dist/mlt/images", "The directory with images"
)
tf.app.flags.DEFINE_string(
    "labels_dir", "./dist/mlt/labels", "The directory with labels"
)

tf.app.flags.DEFINE_string(
    "output_dir", "./dist/mlt/tfrecords", "The target directory with TFRecords"
)

_NUM_SHARDS = 4


def _convert_images(images_dir, labels_dir):
    """Converts the ADE20k dataset into into tfrecord format.
    Args:
      images_dir: The directory with images.
      labels_dir: The directory with labels.
    """

    images_filenames = tf.gfile.Glob(os.path.join(images_dir, "*.jpg"))
    random.shuffle(images_filenames)
    labels_filenames = []
    for f in images_filenames:
        # get the filename without the extension
        basename = os.path.basename(f).split(".")[0]
        labels_filename = os.path.join(labels_dir, basename + ".png")
        labels_filenames.append(labels_filename)

    num_images = len(images_filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_example.ImageReader()

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            f"shard-{shard_id:05}-of-{_NUM_SHARDS:05}.tfrecord",
        )
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write(
                    "\r>> Converting image %d/%d shard %d"
                    % (i + 1, num_images, shard_id)
                )
                sys.stdout.flush()

                image_filename = images_filenames[i]
                image_data = tf.io.gfile.GFile(image_filename, "rb").read()
                height, width = image_reader.read_image_dims(image_data)

                labels_filename = labels_filenames[i]
                labels_data = (
                    tf.io.gfile.GFile(labels_filename, "r").read().split("\n")
                )
                bboxes = []
                for line in labels_data:
                    line = (
                        line.strip("\ufeff").strip("\xef\xbb\xbf").split(",")
                    )
                    bbox = np.asarray(list(map(float, line[:8]))) / (
                        [width * 1.0, height * 1.0] * 4
                    )
                    bboxes.append(bbox)

                example = build_example.labelled_image_to_tfexample(
                    image_data, images_filenames[i], height, width, bboxes
                )
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write("\n")
        sys.stdout.flush()


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)
    _convert_images(FLAGS.train_image_folder, FLAGS.train_image_label_folder)


if __name__ == "__main__":
    tf.app.run()
