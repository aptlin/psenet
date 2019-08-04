import tensorflow as tf
from segmentation_models.losses import dice_loss
from segmentation_models import FPN

from psenet import config


def build_fpn(backbone_name: str, n_kernels=config.KERNEL_NUM):
    return FPN(
        backbone_name=backbone_name,
        input_shape=(None, None, 3),
        classes=n_kernels,
        encoder_weights="imagenet",
        activation="sigmoid",
        pyramid_block_filters=256,
    )


def lr_decay(learning_rate, global_step, decay_steps, decay_rate):
    return tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )


def ohem_single(texts, gt_texts, training_masks):
    has_positive_texts = tf.greater(gt_texts, 0.5)
    has_positive_training_masks = tf.greater(training_masks, 0.5)

    training_masks = tf.cast(training_masks, tf.float32)
    training_masks = tf.expand_dims(training_masks, axis=0)

    positive_texts_num = tf.cast(has_positive_texts, tf.int64)
    positive_texts_num = tf.math.reduce_sum(positive_texts_num)

    positive_throwaways_num = tf.logical_and(
        has_positive_texts, tf.logical_not(has_positive_training_masks)
    )
    positive_throwaways_num = tf.cast(positive_throwaways_num, tf.int64)
    positive_throwaways_num = tf.math.reduce_sum(positive_throwaways_num)

    positive_texts_num -= positive_throwaways_num

    has_net_positive_texts = tf.math.equal(positive_texts_num, 0)

    has_negative_texts = tf.less_equal(gt_texts, 0.5)

    negative_texts_num = tf.cast(has_negative_texts, tf.int64)
    negative_texts_num = tf.math.reduce_sum(negative_texts_num)
    negative_texts_num = tf.math.minimum(
        positive_texts_num * 3, negative_texts_num
    )

    has_net_negative_texts = tf.math.equal(negative_texts_num, 0)

    negative_scores = tf.boolean_mask(texts, has_negative_texts)
    negative_scores = tf.sort(negative_scores, direction="DESCENDING")
    threshold = negative_scores[negative_texts_num - 1]

    selected_mask = tf.logical_and(
        has_positive_training_masks,
        tf.logical_or(tf.greater_equal(texts, threshold), has_positive_texts),
    )

    selected_mask = tf.cast(selected_mask, tf.float32)
    selected_mask = tf.expand_dims(selected_mask, axis=0)
    output = tf.cond(
        tf.logical_or(has_net_positive_texts, has_net_negative_texts),
        training_masks,
        selected_mask,
    )

    return output


def ohem_batch(texts, gt_texts, training_masks):
    texts_count = tf.shape(texts)[0]
    texts_count = tf.cast(texts_count, tf.uint64)

    indices = tf.range(texts_count, dtype=tf.uint64)
    selected_masks = tf.map_fn(
        lambda idx: ohem_single(
            texts[idx, :, :], gt_texts[idx, :, :], training_masks[idx, :, :]
        ),
        indices,
        dtype=tf.float32,
    )

    return selected_masks


def loss(ground_truth, predictions):
    return dice_loss(ground_truth, predictions)


def compute_text_metrics(texts, gt_texts, training_masks, text_metrics):
    texts = tf.where(
        tf.greater(texts, 0.5), tf.ones_like(texts), tf.zeros_like(texts)
    )
    gt_text = gt_texts * training_masks
    text_metrics.update(gt_text, texts)
    return text_metrics.compute_scores()


def compute_kernels_metrics(
    kernels, gt_kernels, training_masks, kernel_metrics
):
    mask = gt_kernels * training_masks

    kernel = kernels[:, -1, :, :]
    kernel = tf.math.sigmoid(kernel)
    kernel = tf.where(
        tf.greater(kernel, 0.5), tf.ones_like(kernel), tf.zeros_like(kernel)
    )
    kernel *= mask
    kernel = tf.cast(kernel, tf.float32)

    gt_kernel = gt_kernels[:, -1, :, :]
    gt_kernel *= mask
    kernel_metrics.update(gt_kernel, kernel)

    return kernel_metrics.compute_scores()
