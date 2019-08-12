import argparse
from psenet import config
from psenet.train import model_fn, build_dataset
import tensorflow as tf


def init_estimator(FLAGS):
    params = tf.contrib.training.HParams(
        kernel_num=7,
        backbone_name=FLAGS.backbone_name,
        encoder_weights="imagenet",
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        learning_rate=FLAGS.learning_rate,
        regularization_weight_decay=FLAGS.regularization_weight_decay,
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
        "--resize-length",
        help="The maximum side length of the resized input images",
        default=config.RESIZE_LENGTH,
        type=int,
    )
    PARSER.add_argument(
        "--crop-size",
        help="The size of the square for a crop",
        default=config.CROP_SIZE,
        type=int,
    )
    PARSER.add_argument(
        "--readers-num",
        help="The number of parallel readers",
        default=config.NUM_READERS,
        type=int,
    )
    PARSER.add_argument(
        "--gpus-num",
        help="The number of GPUs to use",
        default=config.GPUS_NUM,
        type=int,
    )
    PARSER.add_argument(
        "--min-scale",
        help="The minimum kernel scale for pre-processing",
        default=config.LEARNING_RATE_DECAY_FACTOR,
        type=float,
    )
    PARSER.add_argument(
        "--regularization-weight-decay",
        help="The L2 regularization loss for Conv layers",
        default=config.REGULARIZATION_WEIGHT_DECAY,
        type=float,
    )
    PARSER.add_argument(
        "--batch-size",
        help="The batch size for training and evaluation",
        default=config.BATCH_SIZE,
        type=int,
    )
    PARSER.add_argument(
        "--train-steps",
        help="The number of training epochs",
        default=config.N_EPOCHS,
        type=int,
    )
    PARSER.add_argument(
        "--eval-steps",
        help="The number of evaluation_steps epochs",
        default=config.N_EVAL_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--save-checkpoints-secs",
        help="Save checkpoints every this many seconds",
        default=config.SAVE_CHECKPOINTS_SECS,
        type=int,
    )
    PARSER.add_argument(
        "--save-summary-steps",
        help="Save summaries every this many steps",
        default=config.SAVE_SUMMARY_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--warm-ckpt",
        help="The checkpoint to initialize from.",
        default=config.WARM_CHECKPOINT,
        type=str,
    )
    PARSER.add_argument(
        "--eval-start-delay-secs",
        help="Start evaluating after waiting for this many seconds",
        default=config.EVAL_START_DELAY_SECS,
        type=int,
    )
    PARSER.add_argument(
        "--eval-throttle-secs",
        help="Do not re-evaluate unless the last evaluation "
        + "was started at least this many seconds ago",
        default=config.EVAL_THROTTLE_SECS,
        type=int,
    )

    FLAGS, _ = PARSER.parse_known_args()
    save_checkpoint(FLAGS)
