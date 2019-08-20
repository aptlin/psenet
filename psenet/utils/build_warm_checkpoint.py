import argparse

import tensorflow as tf

from psenet import config
from psenet.data import DATASETS, build_input_fn
from psenet.model import model_fn


def init_estimator(FLAGS):
    params = tf.contrib.training.HParams(
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        encoder_weights="imagenet",
        kernel_num=FLAGS.kernel_num,
        learning_rate=FLAGS.learning_rate,
        regularization_weight_decay=FLAGS.regularization_weight_decay,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.warm_ckpt, params=params
    )
    return estimator


def save_checkpoint(FLAGS):
    estimator = init_estimator(FLAGS)
    FLAGS.mode = tf.estimator.ModeKeys.TRAIN
    estimator.train(input_fn=build_input_fn(FLAGS), max_steps=1)


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
    PARSER.add_argument(
        "--dataset",
        help="The dataset to load. Must be one of {}.".format(
            list(DATASETS.keys())
        ),
        default=config.PROCESSED_DATA_LABEL,
        type=str,
    )
    PARSER.add_argument(
        "--augment-training-data",
        help="Whether to augment training data.",
        type=config.str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    PARSER.add_argument(
        "--training-data-dir",
        help="The directory with `images/` and `labels/` for training",
        default=config.TRAINING_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--eval-data-dir",
        help="The directory with `images/` and `labels/` for evaluation",
        default=config.EVAL_DATA_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--batch-size",
        help="The batch size for training and evaluation",
        default=config.BATCH_SIZE,
        type=int,
    )
    PARSER.add_argument(
        "--num_readers",
        help="The number of parallel readers",
        default=config.NUM_READERS,
        type=int,
    )
    PARSER.add_argument(
        "--prefetch",
        help="The number of batches to prefetch",
        default=config.PREFETCH,
        type=int,
    )
    PARSER.add_argument(
        "--min-scale",
        help="The minimum kernel scale for pre-processing",
        default=config.LEARNING_RATE_DECAY_FACTOR,
        type=float,
    )
    PARSER.add_argument(
        "--learning-rate",
        help="The initial learning rate",
        default=config.LEARNING_RATE,
        type=float,
    )
    PARSER.add_argument(
        "--decay-rate",
        help="The learning rate decay factor",
        default=config.LEARNING_RATE_DECAY_FACTOR,
        type=float,
    )
    PARSER.add_argument(
        "--decay-steps",
        help="The number of steps before the learning rate decays",
        default=config.LEARNING_RATE_DECAY_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--regularization-weight-decay",
        help="The L2 regularization loss for Conv layers",
        default=config.REGULARIZATION_WEIGHT_DECAY,
        type=float,
    )
    PARSER.add_argument(
        "--resize-length",
        help="The maximum side length of the resized input images",
        default=config.RESIZE_LENGTH,
        type=int,
    )
    FLAGS, _ = PARSER.parse_known_args()
    save_checkpoint(FLAGS)
