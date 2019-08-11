import psenet.config as config
import tensorflow as tf


def dice_loss(labels, predictions, masks):
    labels_shape = tf.shape(labels)
    labels = tf.reshape(labels, [labels_shape[0], -1])

    predictions_shape = tf.shape(predictions)
    predictions = tf.reshape(predictions, [predictions_shape[0], -1])

    masks_shape = tf.shape(masks)
    masks = tf.reshape(masks, [masks_shape[0], -1])

    labels *= masks
    predictions *= masks

    a = tf.math.reduce_sum(predictions * labels, axis=1)
    b = tf.math.reduce_sum(predictions * predictions, axis=1) + config.EPSILON
    c = tf.math.reduce_sum(labels * labels, axis=1) + config.EPSILON

    d = (2 * a) / (b + c)
    loss = tf.math.reduce_mean(d)
    return 1.0 - loss


def ohem_single(labels, predictions, masks):
    has_positive_texts = tf.greater(labels, 0.5)
    has_positive_masks = tf.greater(masks, 0.5)

    default_mask = tf.cast(masks, tf.float32)
    default_mask = tf.expand_dims(default_mask, axis=0)

    positive_texts_num = tf.cast(has_positive_texts, tf.int64)
    positive_texts_num = tf.math.reduce_sum(positive_texts_num)

    positive_throwaways_num = tf.logical_and(
        has_positive_texts, tf.logical_not(has_positive_masks)
    )
    positive_throwaways_num = tf.cast(positive_throwaways_num, tf.int64)
    positive_throwaways_num = tf.math.reduce_sum(positive_throwaways_num)

    positive_texts_num -= positive_throwaways_num

    has_zero_net_positive_texts = tf.math.equal(positive_texts_num, 0)

    has_negative_texts = tf.logical_not(has_positive_texts)

    negative_texts_num = tf.cast(has_negative_texts, tf.int64)
    negative_texts_num = tf.math.reduce_sum(negative_texts_num)
    negative_texts_num = tf.math.minimum(
        positive_texts_num * 3, negative_texts_num
    )

    has_zero_net_negative_texts = tf.math.equal(negative_texts_num, 0)

    def compute_selected_mask():
        negative_scores = tf.boolean_mask(predictions, has_negative_texts)
        negative_scores = tf.sort(negative_scores, direction="DESCENDING")
        threshold = negative_scores[negative_texts_num - 1]

        selected_mask = tf.logical_and(
            has_positive_masks,
            tf.logical_or(
                tf.greater_equal(predictions, threshold), has_positive_texts
            ),
        )

        selected_mask = tf.cast(selected_mask, tf.float32)
        selected_mask = tf.expand_dims(selected_mask, axis=0)
        return selected_mask

    output = tf.cond(
        tf.logical_or(
            has_zero_net_positive_texts, has_zero_net_negative_texts
        ),
        lambda: default_mask,
        compute_selected_mask,
    )

    return output


def ohem_batch(labels, predictions, masks):
    texts_count = tf.shape(predictions)[0]
    texts_count = tf.cast(texts_count, tf.int64)

    indices = tf.range(texts_count, dtype=tf.int64)
    selected_masks = tf.map_fn(
        lambda idx: ohem_single(
            labels[idx, :, :], predictions[idx, :, :], masks[idx, :, :]
        ),
        indices,
        dtype=tf.float32,
    )
    return selected_masks


def compute_loss(labels, predictions, masks):
    predicted_texts = predictions[:, :, :, 0]
    ground_truth_texts = labels[:, :, :, 0]

    predicted_kernels = predictions[:, :, :, 1:]
    ground_truth_kernels = labels[:, :, :, 1:]

    # compute text loss
    selected_masks = ohem_batch(ground_truth_texts, predicted_texts, masks)
    text_loss = dice_loss(ground_truth_texts, predicted_texts, selected_masks)

    # compute kernel loss
    selected_masks = tf.logical_and(
        tf.greater(predicted_texts, 0.5), tf.greater(masks, 0.5)
    )
    selected_masks = tf.cast(selected_masks, tf.float32)

    indices = tf.range(tf.shape(ground_truth_kernels)[3])

    def compute_kernel_loss(index):
        return dice_loss(
            ground_truth_kernels[:, :, :, index],
            predicted_kernels[:, :, :, index],
            selected_masks,
        )

    kernel_loss = tf.math.reduce_mean(
        tf.map_fn(compute_kernel_loss, indices, dtype=tf.float32)
    )

    current_loss = (
        config.TEXT_LOSS_WEIGHT * text_loss
        + config.KERNELS_LOSS_WEIGHT * kernel_loss
    )

    return text_loss, kernel_loss, current_loss
