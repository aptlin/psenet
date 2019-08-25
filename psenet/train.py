import argparse
import os

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import tf_logging as logging

from psenet import config
from psenet.data import DATASETS, build_input_fn
from psenet.losses import psenet_loss
from psenet.metrics import keras_psenet_metrics
from psenet.model import build_model
from psenet.optimizers import build_optimizer


def build_callbacks(FLAGS):
    checkpoint_prefix = os.path.join(FLAGS.job_dir, "psenet_{epoch}")
    log_dir = os.path.join(FLAGS.job_dir, "logs")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        ),
    ]

    return callbacks


def train(FLAGS):
    strategy = tf.distribute.MirroredStrategy()
    logging.info(
        "Number of replicas in sync: {}".format(strategy.num_replicas_in_sync)
    )

    FLAGS.mode = tf.estimator.ModeKeys.TRAIN
    FLAGS.encoder_weights = "imagenet"

    data = build_input_fn(FLAGS)()
    with strategy.scope():
        model = build_model(FLAGS)
        if FLAGS.use_pretrained:
            model.load_weights(FLAGS.warm_checkpoint, by_name=True)
        model.compile(
            loss=psenet_loss,
            optimizer=build_optimizer(FLAGS),
            metrics=keras_psenet_metrics(),
        )
        model.fit(
            data,
            epochs=FLAGS.num_epochs,
            steps_per_epoch=FLAGS.steps_per_epoch,
            callbacks=build_callbacks(FLAGS),
            verbose=2,
        )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--batch-size",
        help="The batch size for training and evaluation",
        default=config.BATCH_SIZE,
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
        "--dataset",
        help="The dataset to load. Must be one of {}.".format(
            list(DATASETS.keys())
        ),
        default=config.PROCESSED_DATA_LABEL,
        type=str,
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
        "--job-dir",
        help="The model directory",
        default=config.MODEL_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--kernel-num",
        help="The number of output kernels from FPN",
        default=config.KERNEL_NUM,
        type=int,
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
        "--num-readers",
        help="The number of parallel readers",
        default=config.NUM_READERS,
        type=int,
    )
    PARSER.add_argument(
        "--num-gpus",
        help="The number of GPUs to use",
        default=config.GPU_PER_WORKER,
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
        "--regularization-weight-decay",
        help="The L2 regularization loss for Conv layers",
        default=config.REGULARIZATION_WEIGHT_DECAY,
        type=float,
    )
    PARSER.add_argument(
        "--num-epochs",
        help="The number of training epochs",
        default=config.N_EPOCHS,
        type=int,
    )
    PARSER.add_argument(
        "--steps-per-epoch",
        help="The number of steps per epoch",
        default=config.N_SAMPLES,
        type=int,
    )
    PARSER.add_argument(
        "--save-checkpoints-steps",
        help="Save checkpoints every this many steps",
        default=config.SAVE_CHECKPOINTS_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--save-summary-steps",
        help="Save summaries every this many steps",
        default=config.SAVE_SUMMARY_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--warm-checkpoint",
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
    PARSER.add_argument(
        "--distribution-strategy",
        help="The distribution strategy to use. "
        + "Either `mirrored` or `multi-worker-mirrored`.",
        default=config.MIRRORED_STRATEGY,
        type=str,
    )
    PARSER.add_argument(
        "--use-pretrained",
        help="Whether to load the weights from the warm checkpoint.",
        type=config.str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    PARSER.add_argument(
        "--augment-training-data",
        help="Whether to augment training data.",
        type=config.str2bool,
        nargs="?",
        const=True,
        default=True,
    )

    FLAGS, _ = PARSER.parse_known_args()
    tf.compat.v1.logging.set_verbosity("DEBUG")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in range(FLAGS.num_gpus)]
    )
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if FLAGS.num_gpus > 0:
        if tf.test.gpu_device_name():
            logging.info("Default GPU: {}".format(tf.test.gpu_device_name()))
            logging.info(
                "All Devices: {}".format(device_lib.list_local_devices())
            )
        else:
            raise RuntimeError("Failed to find the default GPU.")

    train(FLAGS)
