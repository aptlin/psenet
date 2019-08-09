import argparse
import os

import tensorflow as tf

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
            num_readers=FLAGS.readers_num,
            # should_shuffle=is_training,
            should_shuffle=False,
            should_repeat=True,
            # should_augment=False,
            should_augment=is_training,
        ).build()
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return input_fn


def build_exporter():
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


def build_optimizer(params):
    return tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate,
            decay_steps=params.decay_steps,
            decay_rate=params.decay_rate,
            staircase=True,
        )
    )
    # return tf.keras.optimizers.SGD(
    #     learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
    #         params.learning_rate,
    #         decay_steps=params.decay_steps,
    #         decay_rate=params.decay_rate,
    #         staircase=True,
    #     ),
    #     momentum=config.MOMENTUM,
    # )


def build_model(params, FLAGS):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.float32
    )
    kernels = FPN(
        backbone_name=params.backbone_name,
        input_shape=(None, None, 3),
        classes=params.kernel_num,
        activation="sigmoid",
        weights=None,
        encoder_weights="imagenet",
        encoder_features="default",
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation="concat",
        pyramid_dropout=None,
    )(images)

    def augment(tensors):
        images = tensors[0]
        kernels = tensors[1]
        images_shape = tf.shape(images)
        batch_size = images_shape[0]
        height = images_shape[1]
        width = images_shape[2]
        ones = tf.ones([batch_size, height, width, 1])
        kernels = tf.image.pad_to_bounding_box(kernels, 0, 0, height, width)
        return tf.concat([kernels, ones], axis=-1)

    predictions = tf.keras.layers.Lambda(
        augment,
        output_shape=[None, None, params.kernel_num + 1],
        name=config.KERNELS,
    )([images, kernels])
    psenet = tf.keras.Model(inputs={config.IMAGE: images}, outputs=predictions)

    psenet.compile(
        optimizer=build_optimizer(params),
        loss=losses.psenet_loss(FLAGS.kernel_num),
    )

    return psenet


def build_estimator(run_config, FLAGS):
    tf.compat.v1.summary.FileWriterCache.clear()

    params = tf.contrib.training.HParams(
        kernel_num=FLAGS.kernel_num,
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        learning_rate=FLAGS.learning_rate,
    )

    psenet = build_model(params, FLAGS)
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=psenet, model_dir=FLAGS.job_dir, config=run_config
    )

    estimator = tf.contrib.estimator.add_metrics(
        estimator, metrics.build_metrics(FLAGS.kernel_num)
    )

    exporter = build_exporter()

    train_spec = tf.estimator.TrainSpec(
        input_fn=build_dataset(tf.estimator.ModeKeys.TRAIN, FLAGS),
        max_steps=FLAGS.train_steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=build_dataset(tf.estimator.ModeKeys.EVAL, FLAGS),
        exporters=exporter,
        steps=FLAGS.eval_steps,
        start_delay_secs=FLAGS.eval_start_delay_secs,
        throttle_secs=FLAGS.eval_throttle_secs,
    )

    return estimator, train_spec, eval_spec


def train(FLAGS):
    estimator, train_spec, eval_spec = build_estimator(
        tf.estimator.RunConfig(
            model_dir=FLAGS.job_dir,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            save_summary_steps=FLAGS.save_summary_steps,
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        ),
        FLAGS,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
        "--readers-num",
        help="The number of parallel readers",
        default=config.NUM_READERS,
        type=int,
    )
    PARSER.add_argument(
        "--min-scale",
        help="The minimum kernel scale for pre-processing",
        default=config.LEARNING_RATE_DECAY_FACTOR,
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
        "--keep-checkpoint-every-n-hours",
        help="Number of hours between each checkpoint to be saved",
        default=config.KEEP_CHECKPOINT_EVERY_N_HOURS,
        type=int,
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
    train(FLAGS)
