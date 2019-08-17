"""Converts the ICDAR MLT 2019 data to PSENet features and labels."""

import argparse
import math
import random

from glob import glob
import numpy as np
from functools import partial
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from psenet import config
from psenet.data import preprocess
from psenet.utils.examples import int64_list_feature, float_list_feature
from psenet.utils.readers import ImageReader
from psenet.backbones.factory import Backbones
import cv2

_NUM_SHARDS = 256


def build_processed_example(
    image,
    texts,
    bboxes,
    kernel_num=config.KERNEL_NUM,
    min_scale=config.MIN_SCALE,
    backbone_name=config.BACKBONE_NAME,
    resize_length=config.RESIZE_LENGTH,
):
    tags = "".join(
        map(lambda text: str(int(not text.startswith("###"))), texts)
    )
    assert len(tags) == len(bboxes)

    preprocessing_fn = Backbones.get_preprocessing(backbone_name)

    image = preprocess.scale(image)
    image = preprocessing_fn(image)
    height, width = image.shape[:2]
    text_score = np.zeros([height, width], dtype="uint8")
    mask = np.ones([height, width], dtype="uint8")
    bboxes = np.asarray(bboxes)
    bboxes_count, num_points = bboxes.shape[:2]
    if bboxes_count > 0:
        bboxes = np.reshape(
            bboxes * ([width, height] * 4),
            (bboxes_count, int(num_points / 2), 2),
        ).astype("int32")
        for i in range(bboxes_count):
            cv2.drawContours(text_score, [bboxes[i]], -1, 1, -1)
            if tags[i] == "0":
                cv2.drawContours(mask, [bboxes[i]], -1, 0, -1)

    kernels = []
    for i in range(1, kernel_num):
        rate = 1.0 - (1.0 - min_scale) / (kernel_num - 1) * i
        kernel = np.zeros([height, width], dtype="uint8")
        kernel_bboxes = preprocess.shrink(bboxes, rate)
        for i in range(bboxes_count):
            cv2.drawContours(kernel, [kernel_bboxes[i]], -1, 1, -1)
        kernels.append(kernel)

    text_score = np.expand_dims(text_score, axis=0).astype("float32")
    kernels = np.asarray(kernels, dtype="float32")
    label = np.concatenate([text_score, kernels], axis=0)
    label = np.transpose(label, [1, 2, 0])

    mask = mask.astype("float32")

    assert preprocess.check_numpy_image_validity(
        {config.IMAGE: image}
    ), "Got an invalid image shape {}".format(image.shape)

    image = np.reshape(image, [-1])
    mask = np.reshape(mask, [-1])
    label = np.reshape(label, [-1])

    return height, width, image, mask, label


def _convert_shard(shard_id, target_dir, images_filenames, labels_filenames):
    import tensorflow as tf
    from pathlib import Path

    num_images = len(images_filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = ImageReader()
    output_filename = str(
        Path(
            target_dir,
            f"shard-{(shard_id + 1):05}-of-{_NUM_SHARDS:05}.tfrecord",
        )
    )
    if shard_id * num_per_shard < num_images:
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                image_filename = images_filenames[i]
                image_format = Path(image_filename).name.split(".")[1]
                image_data = tf.io.gfile.GFile(image_filename, "rb").read()
                image = image_reader.decode_image(
                    image_data, image_format, channels=3
                )
                height, width = image.shape[:2]

                labels_filename = labels_filenames[i]
                labels_data = (
                    tf.io.gfile.GFile(labels_filename, "r").read().split("\n")
                )
                bboxes = []
                texts = []
                for line in labels_data:
                    line = (
                        line.strip("\ufeff").strip("\xef\xbb\xbf").split(",")
                    )
                    if len(line) > 8:
                        bbox = np.asarray(list(map(float, line[:8]))) / (
                            [width * 1.0, height * 1.0] * 4
                        )
                        bboxes.append(bbox)
                        text_datum = line[9]
                        texts.append(text_datum)

                height, width, image, mask, label = build_processed_example(
                    image, texts, bboxes
                )
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "height": int64_list_feature(height),
                            "width": int64_list_feature(width),
                            "features/image": float_list_feature(image),
                            "features/mask": float_list_feature(mask),
                            "labels": float_list_feature(label),
                        }
                    )
                )
                tfrecord_writer.write(example.SerializeToString())


def _convert_images(data_dir, target_dir):
    """Converts the ADE20k dataset into into tfrecord format.
    """

    Path(target_dir).mkdir(parents=True)
    images_filenames = glob(
        str(Path(data_dir, config.IMAGES_DIR, "*.jpg"))
    ) + glob(str(Path(data_dir, config.IMAGES_DIR, "*.png")))
    random.shuffle(images_filenames)
    labels_filenames = []
    for f in images_filenames:
        basename = Path(f).name.split(".")[0]
        labels_filename = str(
            Path(data_dir, config.LABELS_DIR, basename + ".txt")
        )
        labels_filenames.append(labels_filename)

    pool = Pool()
    results = list(
        tqdm(
            pool.imap(
                partial(
                    _convert_shard,
                    target_dir=target_dir,
                    images_filenames=images_filenames,
                    labels_filenames=labels_filenames,
                ),
                range(_NUM_SHARDS),
            ),
            total=_NUM_SHARDS,
        )
    )
    pool.close()
    pool.join()
    print(results)


def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--training-data-dir",
        help="The directory with `images/` and `labels/` for training",
        default=config.RAW_TRAINING_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--eval-data-dir",
        help="The directory with `images/` and `labels/` for evaluation",
        default=config.RAW_EVAL_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--output-dir",
        help="The target directory containing"
        + "train/ and eval/ subdirectories with TFRecords",
        default=config.BASE_DATA_DIR,
        type=str,
    )
    FLAGS, _ = PARSER.parse_known_args()

    train_target_dir = Path(FLAGS.output_dir, "train")
    eval_target_dir = Path(FLAGS.output_dir, "eval")

    _convert_images(FLAGS.training_data_dir, train_target_dir)
    _convert_images(FLAGS.eval_data_dir, eval_target_dir)


if __name__ == "__main__":
    main()
