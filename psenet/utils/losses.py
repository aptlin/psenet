from psenet import config
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

    print_labels = tf.print("masked labels nonzero:", tf.count_nonzero(labels))
    print_predictions = tf.print(
        "masked predictions nonzero:", tf.count_nonzero(predictions)
    )

    a = tf.math.reduce_sum(predictions * labels, axis=1)
    b = tf.math.reduce_sum(predictions * predictions, axis=1) + config.EPSILON
    c = tf.math.reduce_sum(labels * labels, axis=1) + config.EPSILON

    print_a = tf.print("a:", a)
    print_b = tf.print("b:", b)
    print_c = tf.print("c:", c)

    with tf.control_dependencies(
        [print_labels, print_predictions, print_a, print_b, print_c]
    ):
        d = (2 * a) / (b + c)
        loss = tf.math.reduce_mean(d)
        return 1 - loss


def ohem_single(labels, predictions, masks):
    has_positive_texts = tf.greater(labels, 0.5)
    has_positive_training_masks = tf.greater(masks, 0.5)

    masks = tf.cast(masks, tf.float32)
    # masks_shape = tf.shape(masks)
    # masks = tf.reshape(masks, [1, masks_shape[0], masks_shape[1]])

    positive_texts_num = tf.cast(has_positive_texts, tf.int64)
    positive_texts_num = tf.math.reduce_sum(positive_texts_num)

    positive_throwaways_num = tf.logical_and(
        has_positive_texts, tf.logical_not(has_positive_training_masks)
    )
    positive_throwaways_num = tf.cast(positive_throwaways_num, tf.int64)
    positive_throwaways_num = tf.math.reduce_sum(positive_throwaways_num)

    positive_texts_num -= positive_throwaways_num

    has_zero_net_positive_texts = tf.math.equal(positive_texts_num, 0)

    has_negative_texts = tf.less_equal(labels, 0.5)

    negative_texts_num = tf.cast(has_negative_texts, tf.int64)
    negative_texts_num = tf.math.reduce_sum(negative_texts_num)
    negative_texts_num = tf.math.minimum(
        positive_texts_num * 3, negative_texts_num
    )

    has_zero_net_negative_texts = tf.math.equal(negative_texts_num, 0)

    negative_scores = tf.boolean_mask(predictions, has_negative_texts)
    negative_scores = tf.sort(negative_scores, direction="DESCENDING")
    threshold = negative_scores[negative_texts_num - 1]

    selected_mask = tf.logical_and(
        has_positive_training_masks,
        tf.logical_or(
            tf.greater_equal(predictions, threshold), has_positive_texts
        ),
    )

    selected_mask = tf.cast(selected_mask, tf.float32)
    # selected_mask_shape = tf.shape(selected_mask)
    # selected_mask = tf.reshape(
    #     masks, [1, selected_mask_shape[0], selected_mask_shape[1]]
    # )
    output = tf.cond(
        tf.logical_or(
            has_zero_net_positive_texts, has_zero_net_negative_texts
        ),
        lambda: masks,
        lambda: selected_mask,
    )

    return output


def ohem_batch(labels, predictions, masks):
    texts_count = tf.shape(predictions)[0]
    texts_count = tf.cast(texts_count, tf.int64)

    indices = tf.range(texts_count, dtype=tf.int64)
    selected_masks = tf.map_fn(
        lambda idx: ohem_single(
            predictions[idx, :, :], labels[idx, :, :], masks[idx, :, :]
        ),
        indices,
        dtype=tf.float32,
    )

    return selected_masks


def psenet_loss(n_kernels):
    def loss(gt_labels, pred_labels):
        texts = pred_labels[:, :, :, 0]
        text_scores = tf.math.sigmoid(texts)
        kernels = pred_labels[:, :, :, 1:n_kernels]

        gt_texts = gt_labels[:, :, :, 0]
        gt_kernels = gt_labels[:, :, :, 1:n_kernels]
        masks = gt_labels[:, :, :, n_kernels]

        selected_masks = ohem_batch(gt_texts, texts, masks)

        text_loss = dice_loss(gt_texts, text_scores, selected_masks)
        print_text_scores = tf.print("text scores:", tf.shape(text_scores))
        print_kernels = tf.print("kernels:", tf.shape(kernels))
        print_orig_masks = tf.print("original masks:", tf.shape(masks))
        print_gt_text_scores = tf.print("gt text scores:", tf.shape(gt_texts))
        print_kernels = tf.print("gt kernels:", tf.shape(gt_kernels))
        print_selected_mask = tf.print(
            "selected_masks nonzero:", tf.count_nonzero(selected_masks)
        )
        print_text_loss = tf.print("text loss:", tf.shape(text_loss))

        with tf.control_dependencies(
            [
                print_text_scores,
                print_kernels,
                print_orig_masks,
                print_gt_text_scores,
                print_kernels,
                print_selected_mask,
                print_text_loss,
            ]
        ):
            kernels_losses = []
        selected_masks = tf.logical_and(
            tf.greater(text_scores, 0.5), tf.greater(masks, 0.5)
        )
        selected_masks = tf.cast(selected_masks, tf.float32)
        indices = tf.range(tf.shape(gt_labels)[3])

        def compute_kernel_loss(index):
            kernel_score = tf.math.sigmoid(kernels[:, :, :, index])
            return dice_loss(
                gt_kernels[:, :, :, index], kernel_score, selected_masks
            )

        kernels_loss = tf.math.reduce_sum(
            tf.map_fn(compute_kernel_loss, indices, dtype=tf.float32)
        )
        kernels_loss = tf.math.reduce_mean(kernels_losses)

        current_loss = (
            config.TEXT_LOSS_WEIGHT * text_loss
            + config.KERNELS_LOSS_WEIGHT * kernels_loss
        ) + config.EPSILON
        print_masks = tf.print("masks:", selected_masks)
        print_loss = tf.print("loss:", selected_masks)
        with tf.control_dependencies([print_masks, print_loss]):
            return current_loss

    return loss
