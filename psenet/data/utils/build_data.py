"""Converts the ICDAR MLT 2019 data to TFRecords."""

from absl import app
from absl import flags
import threading
from tqdm import tqdm
import math
import os
import random
import build_example
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "images_dir", "./dist/mlt/images", "The directory with images"
)
flags.DEFINE_string(
    "labels_dir", "./dist/mlt/labels", "The directory with labels"
)

flags.DEFINE_string(
    "output_dir", "./dist/mlt/tfrecords", "The target directory with TFRecords"
)

_NUM_SHARDS = 4


def _convert_shard(shard_id, jpeg_images_filenames, labels_filenames):
    num_images = len(jpeg_images_filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_example.ImageReader()
    output_filename = os.path.join(
        FLAGS.output_dir,
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


def _convert_images(images_dir, labels_dir):
    """Converts the ADE20k dataset into into tfrecord format.
    Args:
      images_dir: The directory with images.
      labels_dir: The directory with labels.
    """

    jpeg_images_filenames = tf.io.gfile.glob(
        os.path.join(images_dir, "*.jpg")
    ) + tf.io.gfile.glob(os.path.join(images_dir, "*.png"))
    random.shuffle(jpeg_images_filenames)
    labels_filenames = []
    for f in jpeg_images_filenames:
        # get the filename without the extension
        basename = os.path.basename(f).split(".")[0]
        labels_filename = os.path.join(labels_dir, basename + ".txt")
        labels_filenames.append(labels_filename)
    coord = tf.train.Coordinator()
    threads = []
    for shard_id in range(_NUM_SHARDS):
        thread = threading.Thread(
            target=_convert_shard,
            args=(shard_id, jpeg_images_filenames, labels_filenames),
        )
        thread.start()
        threads.append(thread)
    coord.join(threads)


def main(unused_argv):
    tf.io.gfile.makedirs(FLAGS.output_dir)
    _convert_images(FLAGS.images_dir, FLAGS.labels_dir)


if __name__ == "__main__":
    app.run(main)
