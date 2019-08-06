import tensorflow as tf
from absl import app, flags

from psenet import config
from psenet.data.generator import Dataset
from psenet.utils.layers import feature_pyramid_network
from psenet.utils.losses import psenet_loss
from psenet.utils.metrics import (
    OverallAccuracy,
    MeanAccuracy,
    MeanIOU,
    FrequencyWeightedAccuracy,
    filter_kernels,
    filter_texts,
)

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
            should_augment=is_training,
        ).build()
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return input_fn


def build_exporter():
    def serving_input_fn():
        features = {
            config.IMAGE: tf.placeholder(
                dtype=tf.uint8, shape=[None, None, None, 3]
            ),
            config.MASK: tf.placeholder(
                dtype=tf.uint8, shape=[None, None, None]
            ),
        }
        receiver_tensors = {
            config.IMAGE: tf.placeholder(
                dtype=tf.uint8, shape=[None, None, None, 3]
            ),
            config.MASK: tf.placeholder(
                dtype=tf.uint8, shape=[None, None, None]
            ),
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors
        )

    return tf.estimator.LatestExporter(
        name="exporter", serving_input_receiver_fn=serving_input_fn
    )


def build_optimizer(params):
    return tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate,
            decay_steps=params.decay_steps,
            decay_rate=params.decay_rate,
            staircase=True,
        )
    )


def build_model(params):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.uint8
    )
    masks = tf.keras.Input(
        shape=[None, None], name=config.MASK, dtype=tf.uint8
    )
    predictions = feature_pyramid_network(params)(images)
    psenet = tf.keras.Model(
        inputs={config.IMAGE: images, config.MASK: masks}, outputs=predictions
    )

    text_metrics_label = "Texts"

    def text_overall_accuracy(labels, predictions):
        return OverallAccuracy(text_metrics_label)(
            *filter_texts(predictions[:, :, :, 0], labels[:, :, :, 0])
        )

    def text_mean_accuracy(labels, predictions):
        return MeanAccuracy(text_metrics_label)(
            *filter_texts(predictions[:, :, :, 0], labels[:, :, :, 0])
        )

    def text_mean_iou(labels, predictions):
        return MeanIOU(text_metrics_label)(
            *filter_texts(predictions[:, :, :, 0], labels[:, :, :, 0])
        )

    def text_fw_accuracy(labels, predictions):
        return FrequencyWeightedAccuracy(text_metrics_label)(
            *filter_texts(predictions[:, :, :, 0], labels[:, :, :, 0])
        )

    kernel_metrics_label = "Kernels"

    def kernel_overall_accuracy(labels, predictions):
        return OverallAccuracy(kernel_metrics_label)(
            *filter_kernels(predictions[:, :, :, 1:], labels[:, :, :, 1:])
        )

    def kernel_mean_accuracy(labels, predictions):
        return MeanAccuracy(kernel_metrics_label)(
            *filter_kernels(predictions[:, :, :, 1:], labels[:, :, :, 1:])
        )

    def kernel_mean_iou(labels, predictions):
        return MeanIOU(kernel_metrics_label)(
            *filter_kernels(predictions[:, :, :, 1:], labels[:, :, :, 1:])
        )

    def kernel_fw_accuracy(labels, predictions):
        return FrequencyWeightedAccuracy(kernel_metrics_label)(
            *filter_kernels(predictions[:, :, :, 1:], labels[:, :, :, 1:])
        )

    psenet.compile(
        optimizer=build_optimizer(params),
        loss=psenet_loss(masks),
        metrics=[
            text_overall_accuracy,
            text_mean_accuracy,
            text_mean_iou,
            text_fw_accuracy,
            kernel_overall_accuracy,
            kernel_mean_accuracy,
            kernel_mean_iou,
            kernel_fw_accuracy,
        ],
    )
    return psenet


def build_estimator(run_config):
    tf.summary.FileWriterCache.clear()

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
    )

    return estimator, train_spec, eval_spec


def train(argv):
    estimator, train_spec, eval_spec = build_estimator(
        tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=config.SAVE_CHECKPOINTS_STEPS,
            save_summary_steps=config.SAVE_SUMMARY_STEPS,
        )
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    app.run(train)
