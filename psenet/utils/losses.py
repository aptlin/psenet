from psenet import config
import tensorflow as tf
from segmentation_models.losses import dice_loss


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

    has_zero_net_positive_texts = tf.math.equal(positive_texts_num, 0)

    has_negative_texts = tf.less_equal(gt_texts, 0.5)

    negative_texts_num = tf.cast(has_negative_texts, tf.int64)
    negative_texts_num = tf.math.reduce_sum(negative_texts_num)
    negative_texts_num = tf.math.minimum(
        positive_texts_num * 3, negative_texts_num
    )

    has_zero_net_negative_texts = tf.math.equal(negative_texts_num, 0)

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
        tf.logical_or(
            has_zero_net_positive_texts, has_zero_net_negative_texts
        ),
        lambda: training_masks,
        lambda: selected_mask,
    )

    return output


def ohem_batch(texts, gt_texts, training_masks):
    texts_count = tf.shape(texts)[0]
    texts_count = tf.cast(texts_count, tf.int64)

    indices = tf.range(texts_count, dtype=tf.int64)
    selected_masks = tf.map_fn(
        lambda idx: ohem_single(
            texts[idx, :, :], gt_texts[idx, :, :], training_masks[idx, :, :]
        ),
        indices,
        dtype=tf.float32,
    )

    return selected_masks


def psenet_loss(masks):
    def loss(gt_labels, pred_labels):
        text_scores = tf.math.sigmoid(pred_labels[:, :, :, 0])
        kernels = pred_labels[:, :, :, 1:]

        gt_texts = gt_labels[:, :, :, 0]
        gt_kernels = gt_labels[:, :, :, 1:]

        selected_masks = ohem_batch(text_scores, gt_texts, masks)

        text_loss = dice_loss(
            gt_texts * selected_masks, text_scores * selected_masks
        )

        kernels_losses = []
        selected_masks = tf.logical_and(
            tf.greater(text_scores, 0.5), tf.greater(masks, 0.5)
        )
        selected_masks = tf.cast(selected_masks, tf.float32)
        indices = tf.range(tf.shape(gt_labels)[3])

        def compute_kernel_loss(index):
            kernel_score = tf.math.sigmoid(kernels[:, :, :, index])
            return dice_loss(
                gt_kernels[:, :, :, index] * selected_masks,
                kernel_score * selected_masks,
            )

        kernels_loss = tf.math.reduce_sum(
            tf.map_fn(compute_kernel_loss, indices, dtype=tf.float32)
        )
        kernels_loss = tf.math.reduce_mean(kernels_losses)

        current_loss = (
            config.TEXT_LOSS_WEIGHT * text_loss
            + config.KERNELS_LOSS_WEIGHT * kernels_loss
        )
        return current_loss

    return loss
