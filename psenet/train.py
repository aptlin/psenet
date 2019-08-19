import argparse
import copy
import json
import os

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import tf_logging as logging

from psenet import config
from psenet.data import DATASETS, build_input_fn
from psenet.eval import build_eval_exporter
from psenet.model import model_fn


def train_and_evaluate(FLAGS):

    TRAIN_FLAGS = copy.deepcopy(FLAGS)
    TRAIN_FLAGS.mode = tf.estimator.ModeKeys.TRAIN
    train_spec = tf.estimator.TrainSpec(
        input_fn=build_input_fn(TRAIN_FLAGS), max_steps=FLAGS.train_steps
    )

    EVAL_FLAGS = copy.deepcopy(FLAGS)
    EVAL_FLAGS.mode = tf.estimator.ModeKeys.EVAL
    eval_exporter = build_eval_exporter()
    eval_spec = tf.estimator.EvalSpec(
        input_fn=build_input_fn(EVAL_FLAGS),
        exporters=eval_exporter,
        steps=FLAGS.eval_steps,
        start_delay_secs=FLAGS.eval_start_delay_secs,
        throttle_secs=FLAGS.eval_throttle_secs,
    )

    if FLAGS.distribution_strategy == config.MIRRORED_STRATEGY:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    params = tf.contrib.training.HParams(
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        encoder_weights=None,
        kernel_num=FLAGS.kernel_num,
        learning_rate=FLAGS.learning_rate,
        regularization_weight_decay=FLAGS.regularization_weight_decay,
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.job_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_summary_steps=FLAGS.save_summary_steps,
        train_distribute=strategy,
        eval_distribute=strategy,
        keep_checkpoint_max=10,
        session_config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            device_count={"GPU": FLAGS.gpu_per_worker},
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                visible_device_list=",".join(
                    [str(i) for i in range(FLAGS.gpu_per_worker)]
                ),
            ),
        ),
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.job_dir,
        config=run_config,
        params=params,
        warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.warm_ckpt
        ),
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
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
        "--num_readers",
        help="The number of parallel readers",
        default=config.NUM_READERS,
        type=int,
    )
    PARSER.add_argument(
        "--gpu-per-worker",
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
    PARSER.add_argument(
        "--distribution-strategy",
        help="The distribution strategy to use. "
        + "Either `mirrored` or `multi-worker-mirrored`.",
        default=config.MIRRORED_STRATEGY,
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
        [str(i) for i in range(FLAGS.gpu_per_worker)]
    )
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if FLAGS.distribution_strategy == config.MULTIWORKER_MIRRORED_STRATEGY:
        tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
        if (
            "task" in tf_config
            and "type" in tf_config["task"]
            and tf_config["task"]["type"] == "master"
        ):
            tf_config["task"]["type"] = "chief"
        if "cluster" in tf_config and "master" in tf_config["cluster"]:
            master_cluster = tf_config["cluster"].pop("master")
            tf_config["cluster"]["chief"] = master_cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        logging.info(
            "Changed the config to {}".format(
                json.loads(os.environ.get("TF_CONFIG", "{}"))
            )
        )
    elif FLAGS.distribution_strategy != config.MIRRORED_STRATEGY:
        raise ValueError("Got an unexpected distribution strategy, aborting.")

    if FLAGS.gpu_per_worker > 0:
        if tf.test.gpu_device_name():
            logging.info("Default GPU: {}".format(tf.test.gpu_device_name()))
            logging.info(
                "All Devices: {}".format(device_lib.list_local_devices())
            )
        else:
            raise RuntimeError("Failed to find the default GPU.")

    train_and_evaluate(FLAGS)
