from psenet import config
import tensorflow as tf
from psenet.nets.fpn import FPN
import psenet.metrics as metrics
import psenet.losses as losses


def build_model(params):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.float32
    )
    weight_decay = None
    if "regularization_weight_decay" in params:
        weight_decay = params.regularization_weight_decay

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
        weight_decay=weight_decay,
    )(images)

    return tf.keras.Model(inputs={config.IMAGE: images}, outputs=kernels)


def build_optimizer(params):
    return tf.compat.v1.train.MomentumOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params.learning_rate,
            global_step=tf.compat.v1.train.get_or_create_global_step(),
            decay_steps=params.decay_steps,
            decay_rate=params.decay_rate,
            staircase=True,
        ),
        momentum=config.MOMENTUM,
    )


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        model = build_model(params)
        image = features[config.IMAGE]
        predictions = model(image, training=False)
        predictions = {config.KERNELS: predictions}
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                "detect": tf.estimator.export.PredictOutput(predictions)
            },
        )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        model = build_model(params)
        optimizer = build_optimizer(params)

        image = features[config.IMAGE]
        predictions = model(image, training=True)
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
                total_loss, tf.compat.v1.train.get_or_create_global_step()
            ),
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
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
    else:
        raise ValueError(
            "The mode {} is not supported, aborting.".format(mode)
        )
