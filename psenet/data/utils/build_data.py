"""Converts the ICDAR MLT 2019 data to TFRecords."""

import math
import os
import random
import threading

import numpy as np
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

import build_example
from psenet import config

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "training_data_dir",
    config.RAW_TRAINING_DATA_DIR,
    "The directory with images and labels for training",
)
flags.DEFINE_string(
    "eval_data_dir",
    config.RAW_EVAL_DATA_DIR,
    "The directory with images and labels for evaluation",
)

flags.DEFINE_string(
    "output_dir", "./dist/mlt/tfrecords", "The target directory with TFRecords"
)

_NUM_SHARDS = 4


def _convert_shard(stage, shard_id, jpeg_images_filenames, labels_filenames):
    num_images = len(jpeg_images_filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_example.ImageReader()
    output_filename = os.path.join(
        FLAGS.output_dir,
        stage,
        f"shard-{(shard_id + 1):05}-of-{_NUM_SHARDS:05}.tfrecord",
    )
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        for i in tqdm(range(start_idx, end_idx)):
            image_filename = jpeg_images_filenames[i]
            image_format = os.path.basename(image_filename).split(".")[1]
            image_data = tf.io.gfile.GFile(image_filename, "rb").read()
            height, width = image_reader.read_image_dims(
                image_data, image_format
            )

            labels_filename = labels_filenames[i]
            labels_data = (
                tf.io.gfile.GFile(labels_filename, "r").read().split("\n")
            )
            bboxes = []
            text_data = []
            for line in labels_data:
                line = line.strip("\ufeff").strip("\xef\xbb\xbf").split(",")
                if len(line) > 8:
                    bbox = np.asarray(list(map(float, line[:8]))) / (
                        [width * 1.0, height * 1.0] * 4
                    )
                    bboxes.extend(bbox)
                    text_datum = line[9]
                    text_data.append(text_datum)

            example = build_example.labelled_image_to_tfexample(
                image_data,
                text_data,
                jpeg_images_filenames[i],
                height,
                width,
                bboxes,
            )
            tfrecord_writer.write(example.SerializeToString())


def _convert_images(data_dir, stage):
    """Converts the ADE20k dataset into into tfrecord format.
    Args:
      images_dir: The directory with images.
      labels_dir: The directory with labels.
    """

    jpeg_images_filenames = tf.io.gfile.glob(
        os.path.join(data_dir, config.IMAGES_DIR, "*.jpg")
    ) + tf.io.gfile.glob(os.path.join(data_dir, config.IMAGES_DIR, "*.png"))
    random.shuffle(jpeg_images_filenames)
    labels_filenames = []
    for f in jpeg_images_filenames:
        basename = os.path.basename(f).split(".")[0]
        labels_filename = os.path.join(
            data_dir, config.LABELS_DIR, basename + ".txt"
        )
        labels_filenames.append(labels_filename)
    coord = tf.train.Coordinator()
    threads = []
    for shard_id in range(_NUM_SHARDS):
        thread = threading.Thread(
            target=_convert_shard,
            args=(stage, shard_id, jpeg_images_filenames, labels_filenames),
        )
        thread.start()
        threads.append(thread)
    coord.join(threads)
    print("Done!")


def main(unused_argv):
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_dir, "train"))
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_dir, "eval"))
    _convert_images(FLAGS.training_data_dir, "train")
    _convert_images(FLAGS.eval_data_dir, "eval")


if __name__ == "__main__":
    app.run(main)
