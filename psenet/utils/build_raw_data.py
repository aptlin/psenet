"""Converts the ICDAR MLT 2019 data to TFRecords."""

import argparse
import math
import os
import random
import threading

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import build_example
from psenet import config


_NUM_SHARDS = 8


def _convert_shard(
    target_dir, shard_id, jpeg_images_filenames, labels_filenames
):
    num_images = len(jpeg_images_filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_example.ImageReader()
    output_filename = os.path.join(
        target_dir, f"shard-{(shard_id + 1):05}-of-{_NUM_SHARDS:05}.tfrecord"
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


def _convert_images(data_dir, target_dir):
    """Converts the ADE20k dataset into into tfrecord format.
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
            args=(
                target_dir,
                shard_id,
                jpeg_images_filenames,
                labels_filenames,
            ),
        )
        thread.start()
        threads.append(thread)
    coord.join(threads)
    print("Done!")


def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--training-data-dir",
        help="The directory with `images/` and `labels/` for training",
        default=config.TRAINING_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--eval-data-dir",
        help="The directory with `images/` and `labels/` for evaluation",
        default=config.TRAINING_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--output-dir",
        help="The target directorycontaining"
        + "train/ and eval/ subdirectories with TFRecords",
        default=config.TRAINING_DATA_DIR,
        type=str,
    )
    FLAGS, _ = PARSER.parse_known_args()

    train_target_dir = os.path.join(FLAGS.output_dir, "train")
    eval_target_dir = os.path.join(FLAGS.output_dir, "eval")

    tf.io.gfile.makedirs(train_target_dir)
    tf.io.gfile.makedirs(eval_target_dir)

    _convert_images(FLAGS.training_data_dir, train_target_dir)
    _convert_images(FLAGS.eval_data_dir, eval_target_dir)


if __name__ == "__main__":
    main()
