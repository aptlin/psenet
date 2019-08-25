import argparse
import os

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import tf_logging as logging

from psenet import config
from psenet.data import DATASETS, build_input_fn
from psenet.losses import psenet_loss
from psenet.metrics import keras_psenet_metrics
from psenet.optimizers import build_optimizer


def build_eval_exporter():
    def serving_input_fn():
        features = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            )
        }
        receiver_tensors = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            )
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors
        )

    return tf.estimator.LatestExporter(
        name="exporter", serving_input_receiver_fn=serving_input_fn
    )


def build_callbacks(FLAGS):
    log_dir = os.path.join(FLAGS.job_dir, "logs")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
    ]

    return callbacks


def evaluate(FLAGS):
    FLAGS.mode = tf.estimator.ModeKeys.EVAL
    FLAGS.encoder_weights = "imagenet"

    data = build_input_fn(FLAGS)()
    model = tf.keras.experimental.load_from_saved_model(FLAGS.saved_model)
    model.compile(
        loss=psenet_loss,
        optimizer=build_optimizer(FLAGS),
        metrics=keras_psenet_metrics(),
    )
    model.evaluate(
        data, steps=FLAGS.num_steps, callbacks=build_callbacks(FLAGS)
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
        help="The type of dataset to load. Must be one of {}.".format(
            list(DATASETS.keys())
        ),
        default=config.PROCESSED_DATA_LABEL,
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
        "--num-steps",
        help="The number of evaluation steps",
        default=config.N_EVAL_STEPS,
        type=int,
    )
    PARSER.add_argument(
        "--saved_model",
        help="The saved model to load.",
        default=config.SAVED_MODEL_DIR,
        type=str,
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

    evaluate(FLAGS)
