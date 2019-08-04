from functools import partial

import tensorflow as tf

from psenet import config
from psenet.utils import training
from psenet.utils.metrics import RunningScore


def build_model(features, labels, mode, params):
    text_metrics = RunningScore(2, "Texts")
    kernel_metrics = RunningScore(2, "Kernels")
    images = features[config.IMAGE_NAME]

    fpn = training.build_fpn(params.backbone_name, params.n_kernels)
    predictions = fpn(images)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        text_scores = tf.math.sigmoid(predictions[:, 0, :, :])
        kernels = predictions[:, 1:, :, :]

        training_masks = features[config.TRAINING_MASK]
        gt_texts = labels[:, 0, :, :]
        gt_kernels = labels[:, 1:, :, :]

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
            kernel_score = tf.math.sigmoid(kernels[:, idx, :, :])
            kernels_losses.append(
                training.loss(
                    gt_kernels[:, idx, :, :] * selected_masks,
                    kernel_score * selected_masks,
                )
            )
        kernels_loss = tf.math.reduce_mean(kernels_losses)

        current_loss = (
            config.TEXT_LOSS_WEIGHT * text_loss
            + config.KERNELS_LOSS_WEIGHT * kernels_loss
        )

        text_results = training.compute_text_metrics(
            text_scores, gt_texts, text_metrics
        )
        kernel_results = training.compute_kernel_metrics(
            kernels, gt_kernels, kernel_metrics
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
