import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.client import device_lib

import psenet.config as config
import psenet.data as data
import psenet.losses as losses
import psenet.metrics as metrics
from psenet.layers.fpn import FPN


def build_dataset(mode, FLAGS):
    def input_fn():
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        dataset_dir = (
            FLAGS.training_data_dir if is_training else FLAGS.eval_data_dir
        )
        dataset = data.Dataset(
            dataset_dir,
            FLAGS.batch_size,
            resize_length=FLAGS.resize_length,
            min_scale=FLAGS.min_scale,
            kernel_num=FLAGS.kernel_num,
            crop_size=FLAGS.crop_size,
            num_readers=FLAGS.readers_num,
            should_shuffle=is_training,
            should_repeat=True,
            # should_augment=False
            should_augment=is_training,
        ).build()
        return dataset

    return input_fn


def build_exporter():
    def serving_input_fn():
        features = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            ),
            config.MASK: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None]
            ),
        }
        receiver_tensors = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            ),
            config.MASK: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None]
            ),
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors
        )

    return tf.estimator.LatestExporter(
        name="exporter", serving_input_receiver_fn=serving_input_fn
    )


def build_optimizer(params):
    # return tf.train.AdamOptimizer(
    #     learning_rate=tf.compat.v1.train.exponential_decay(
    #         learning_rate=params.learning_rate,
    #         global_step=tf.train.get_or_create_global_step(),
    #         decay_steps=params.decay_steps,
    #         decay_rate=params.decay_rate,
    #         staircase=True,
    #     )
    # )
    return tf.train.MomentumOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params.learning_rate,
            global_step=tf.train.get_or_create_global_step(),
            decay_steps=params.decay_steps,
            decay_rate=params.decay_rate,
            staircase=True,
        ),
        momentum=config.MOMENTUM,
    )


def build_model(params):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.float32
    )
    kernels = FPN(
        backbone_name=params.backbone_name,
        input_shape=(None, None, 3),
        classes=params.kernel_num,
        activation="sigmoid",
        weights=None,
        encoder_weights=params.encoder_weights,
        encoder_features="default",
        pyramid_block_filters=256,
        segmentation_filters=128,
        pyramid_use_batchnorm=True,
        pyramid_aggregation="concat",
        pyramid_dropout=None,
        weight_decay=params.regularization_weight_decay,
    )(images)

    return tf.keras.Model(inputs={config.IMAGE: images}, outputs=kernels)


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        model = build_model(params)
        predictions = model(features[config.IMAGE], training=False)
        predictions = {"kernels": predictions}
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                "detect": tf.estimator.export.PredictOutput(predictions)
            },
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        model = build_model(params)
        optimizer = build_optimizer(params)

        predictions = model(features[config.IMAGE], training=True)
        masks = features[config.MASK]
        text_loss, kernel_loss, total_loss = losses.compute_loss(
            labels, predictions, masks
        )
        tf.compat.v1.summary.scalar("text_loss", text_loss, family="losses")
        tf.compat.v1.summary.scalar(
            "kernel_loss", kernel_loss, family="losses"
        )
        tf.compat.v1.summary.scalar("total_loss", total_loss, family="losses")
        computed_metrics = metrics.build_metrics(labels, predictions, masks)
        for metric_name, op in computed_metrics.items():
            tf.compat.v1.summary.scalar(metric_name, op[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=total_loss,
            train_op=optimizer.minimize(
                total_loss, tf.train.get_or_create_global_step()
            ),
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        model = build_model(params)
        predictions = model(features[config.IMAGE], training=False)
        masks = features[config.MASK]
        text_loss, kernel_loss, total_loss = losses.compute_loss(
            labels, predictions, masks
        )
        tf.compat.v1.summary.scalar("text_loss", text_loss, family="losses")
        tf.compat.v1.summary.scalar(
            "kernel_loss", kernel_loss, family="losses"
        )
        tf.compat.v1.summary.scalar("total_loss", total_loss, family="losses")
        computed_metrics = metrics.build_metrics(labels, predictions, masks)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=total_loss,
            eval_metric_ops=computed_metrics,
        )


def train(FLAGS):
    # exporter = build_exporter()

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=build_dataset(tf.estimator.ModeKeys.TRAIN, FLAGS),
    #     max_steps=FLAGS.train_steps,
    # )

    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=build_dataset(tf.estimator.ModeKeys.EVAL, FLAGS),
    #     exporters=exporter,
    #     steps=FLAGS.eval_steps,
    #     start_delay_secs=FLAGS.eval_start_delay_secs,
    #     throttle_secs=FLAGS.eval_throttle_secs,
    # )

    strategy = tf.distribute.MirroredStrategy()

    params = tf.contrib.training.HParams(
        kernel_num=FLAGS.kernel_num,
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        learning_rate=FLAGS.learning_rate,
        encoder_weights=None,
        regularization_weight_decay=FLAGS.regularization_weight_decay,
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.job_dir,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_summary_steps=FLAGS.save_summary_steps,
        train_distribute=strategy,
        eval_distribute=strategy,
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                visible_device_list=",".join(
                    [str(i) for i in range(FLAGS.gpus_num)]
                ),
                allow_growth=True,
            ),
            device_count={"GPU": FLAGS.gpus_num},
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

    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.train(
        input_fn=build_dataset(tf.estimator.ModeKeys.TRAIN, FLAGS),
        max_steps=FLAGS.train_steps,
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
    tf.compat.v1.logging.set_verbosity("DEBUG")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in range(FLAGS.gpus_num)]
    )
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if FLAGS.gpus_num > 0:
        if tf.test.gpu_device_name():
            print("Default GPU: {}".format(tf.test.gpu_device_name()))
            print("All Devices:\n {}".format(device_lib.list_local_devices()))
        else:
            print("Failed to find default GPU.")
            sys.exit(1)

    train(FLAGS)
