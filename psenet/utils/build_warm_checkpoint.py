import argparse

import tensorflow as tf

from psenet import config
from psenet.data import build_dataset
from psenet.model import model_fn


def init_estimator(FLAGS):
    params = tf.contrib.training.HParams(
        kernel_num=FLAGS.kernel_num,
        backbone_name=FLAGS.backbone_name,
        encoder_weights="imagenet",
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.warm_ckpt, params=params
    )
    return estimator


def save_checkpoint(FLAGS):
    estimator = init_estimator(FLAGS)
    estimator.train(
        input_fn=build_dataset(tf.estimator.ModeKeys.TRAIN, FLAGS), max_steps=2
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--kernel-num",
        help="The number of output kernels from FPN",
        default=config.KERNEL_NUM,
        type=int,
    )
    PARSER.add_argument(
        "--backbone-name",
        help="""The name of the FPN backbone. Must be one of the following:
                - 'inceptionresnetv2',
                - 'inceptionv3',
                - 'resnext50',
                - 'resnext101',
                - 'mobilenet',
                - 'mobilenetv2',
                - 'efficientnetb0',
                - 'efficientnetb1',
                - 'efficientnetb2',
                - 'efficientnetb3',
                - 'efficientnetb4',
                - 'efficientnetb5'
        """,
        default=config.BACKBONE_NAME,
        type=str,
    )
    PARSER.add_argument(
        "--warm-ckpt",
        help="The checkpoint to initialize from.",
        default=config.WARM_CHECKPOINT,
        type=str,
    )

    FLAGS, _ = PARSER.parse_known_args()
    save_checkpoint(FLAGS)
