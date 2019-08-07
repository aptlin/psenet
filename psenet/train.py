import tensorflow as tf
from absl import app, flags

from psenet import config
from psenet.data.generator import Dataset
from psenet.utils.layers import feature_pyramid_network
from psenet.utils.losses import psenet_loss
from psenet.utils.metrics import build_metrics

flags.DEFINE_string("model_dir", config.MODEL_DIR, "The model directory")
flags.DEFINE_string(
    "training_data_dir",
    config.TRAINING_DATA_DIR,
    "The directory with images and labels for training",
)
flags.DEFINE_string(
    "eval_data_dir",
    config.EVAL_DATA_DIR,
    "The directory with images and labels for evaluation",
)
flags.DEFINE_integer(
    "n_kernels", config.KERNEL_NUM, "The number of output tensors"
)
flags.DEFINE_string(
    "backbone_name",
    config.BACKBONE_NAME,
    """The name of the FPN backbone. Must be one of the following:
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
)
flags.DEFINE_float(
    "learning_rate", config.LEARNING_RATE, "The initial learning rate"
)
flags.DEFINE_float(
    "decay_rate",
    config.LEARNING_RATE_DECAY_FACTOR,
    "The learning rate decay factor",
)
flags.DEFINE_integer(
    "decay_steps",
    config.LEARNING_RATE_DECAY_STEPS,
    "The number of steps before the learning rate decays",
)
flags.DEFINE_integer(
    "resize_length",
    config.RESIZE_LENGTH,
    "The maximum side length of the resized input images",
)
flags.DEFINE_integer(
    "n_readers", config.NUM_READERS, "The number of parallel readers"
)
flags.DEFINE_float("min_scale", config.MIN_SCALE, "Minimum scale of kernels")
flags.DEFINE_integer(
    "batch_size",
    config.BATCH_SIZE,
    "The batch size for training and evaluation",
)
flags.DEFINE_integer(
    "n_epochs", config.N_EPOCHS, "The number of training epochs"
)
flags.DEFINE_integer(
    "n_eval_steps", config.N_EVAL_STEPS, "The number of evaluation steps"
)
flags.DEFINE_integer(
    "eval_start_delay_secs",
    config.EVAL_START_DELAY_SECS,
    "Start evaluating after waiting for this many seconds",
)
flags.DEFINE_integer(
    "eval_throttle_secs",
    config.EVAL_THROTTLE_SECS,
    "Do not re-evaluate unless the last evaluation "
    + "was started at least this many seconds ago",
)
FLAGS = flags.FLAGS


def build_dataset(mode, dataset_dir):
    def input_fn():
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        dataset = Dataset(
            dataset_dir,
            FLAGS.batch_size,
            resize_length=FLAGS.resize_length,
            min_scale=FLAGS.min_scale,
            kernel_num=FLAGS.n_kernels,
            num_readers=FLAGS.n_readers,
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
            config.IMAGE: tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            )
        }
        receiver_tensors = {
            config.IMAGE: tf.placeholder(
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


def build_model(params):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.float32
    )
    kernels = feature_pyramid_network(params)(images)

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
        output_shape=[None, None, params.n_kernels + 1],
        name=config.KERNELS,
    )([images, kernels])
    psenet = tf.keras.Model(inputs={config.IMAGE: images}, outputs=predictions)

    psenet.compile(
        optimizer=build_optimizer(params), loss=psenet_loss(FLAGS.n_kernels)
    )

    return psenet


def build_estimator(run_config):
    tf.compat.v1.summary.FileWriterCache.clear()

    params = tf.contrib.training.HParams(
        n_kernels=FLAGS.n_kernels,
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        learning_rate=FLAGS.learning_rate,
    )

    psenet = build_model(params)
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=psenet, model_dir=FLAGS.model_dir, config=run_config
    )

    estimator = tf.contrib.estimator.add_metrics(
        estimator, build_metrics(FLAGS.n_kernels)
    )

    exporter = build_exporter()

    train_spec = tf.estimator.TrainSpec(
        input_fn=build_dataset(
            tf.estimator.ModeKeys.TRAIN, FLAGS.training_data_dir
        ),
        max_steps=FLAGS.n_epochs,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=build_dataset(
            tf.estimator.ModeKeys.EVAL, FLAGS.eval_data_dir
        ),
        exporters=exporter,
        steps=FLAGS.n_eval_steps,
        start_delay_secs=FLAGS.eval_start_delay_secs,
        throttle_secs=FLAGS.eval_throttle_secs,
    )

    return estimator, train_spec, eval_spec


def train(argv):
    estimator, train_spec, eval_spec = build_estimator(
        tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=config.SAVE_CHECKPOINTS_SECS,
            save_summary_steps=config.SAVE_SUMMARY_STEPS,
            keep_checkpoint_every_n_hours=config.KEEP_CHECKPOINT_EVERY_N_HOURS,
        )
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    app.run(train)
