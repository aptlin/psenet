from functools import partial

import tensorflow as tf
from absl import app, flags

from psenet import config
from psenet.utils import training
from psenet.utils.metrics import RunningScore
from psenet.data.generator import Dataset

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
    config.NUM_READERS,
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


def build_model(features, labels, mode, params):
    text_metrics = RunningScore(2, "Texts")
    kernel_metrics = RunningScore(2, "Kernels")
    images = features[config.IMAGE]

    fpn = training.build_fpn(params.backbone_name, params.n_kernels)
    predictions = fpn(images)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        text_scores = tf.math.sigmoid(predictions[:, :, :, 0])
        kernels = predictions[:, :, :, 1:]

        gt_texts = labels[:, :, :, 0]
        gt_kernels = labels[:, :, :, 1:]

        training_masks = features[config.TRAINING_MASK]
        selected_masks = training.ohem_batch(
            text_scores, gt_texts, training_masks
        )

        text_loss = training.loss(
            gt_texts * selected_masks, text_scores * selected_masks
        )

        kernels_losses = []
        selected_masks = tf.logical_and(
            tf.greater(text_scores, 0.5), tf.greater(training_masks, 0.5)
        )
        selected_masks = tf.cast(selected_masks, tf.float32)
        for idx in range(params.n_kernels - 1):
            kernel_score = tf.math.sigmoid(kernels[:, :, :, idx])
            kernels_losses.append(
                training.loss(
                    gt_kernels[:, :, :, idx] * selected_masks,
                    kernel_score * selected_masks,
                )
            )
        kernels_loss = tf.math.reduce_mean(kernels_losses)

        current_loss = (
            config.TEXT_LOSS_WEIGHT * text_loss
            + config.KERNELS_LOSS_WEIGHT * kernels_loss
        )

        text_results = training.compute_text_metrics(
            text_scores, gt_texts, training_masks, text_metrics
        )
        kernel_results = training.compute_kernel_metrics(
            kernels, gt_kernels, training_masks, kernel_metrics
        )

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=current_loss,
                eval_metric_ops={**text_results, **kernel_results},
            )
        elif mode == tf.estimator.ModeKeys.TRAIN:
            decay_rate = params.decay_rate
            decay_steps = params.decay_steps
            train_op = tf.contrib.layers.optimize_loss(
                loss=current_loss,
                global_step=tf.train.get_global_step(),
                learning_rate=params.learning_rate,
                optimizer="SGD",
                learning_rate_decay_fn=partial(
                    training.lr_decay,
                    decay_rate=decay_rate,
                    decay_steps=decay_steps,
                ),
                # clip_gradients=params.gradient_clipping_norm,
                summaries=[
                    "learning_rate",
                    "loss",
                    "gradients",
                    "gradient_norm",
                ],
            )
            return tf.estimator.EstimatorSpec(
                mode, loss=current_loss, train_op=train_op
            )

        else:
            raise NotImplementedError("Unknown mode {}".format(mode))


def build_estimator(run_config):
    params = tf.contrib.training.HParams(
        n_kernels=FLAGS.n_kernels,
        backbone_name=FLAGS.backbone_name,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        learning_rate=FLAGS.learning_rate,
    )
    estimator = tf.estimator.Estimator(
        model_fn=build_model, config=run_config, params=params
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=build_dataset(
            tf.estimator.ModeKeys.TRAIN, FLAGS.training_data_dir
        ),
        max_steps=FLAGS.n_epochs,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=build_dataset(tf.estimator.ModeKeys.EVAL, FLAGS.eval_data_dir)
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
