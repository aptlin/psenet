import tensorflow as tf
import psenet.config as config


def filter_texts(labels, predictions, masks):
    text = tf.math.sigmoid(predictions[:, :, :, 0]) * masks
    text = tf.where(
        tf.greater(text, 0.5), tf.ones_like(text), tf.zeros_like(text)
    )
    text = tf.cast(text, tf.int32)

    gt_text = labels[:, :, :, 0] * masks
    gt_text = tf.cast(gt_text, tf.int32)

    return gt_text, text


def filter_kernels(labels, predictions, masks):
    gt_text = labels[:, :, :, 0] * masks

    kernel = tf.math.sigmoid(predictions[:, :, :, -1])
    kernel = tf.where(
        tf.greater(kernel, 0.5), tf.ones_like(kernel), tf.zeros_like(kernel)
    )

    kernel *= gt_text
    kernel = tf.cast(kernel, tf.float32)

    gt_kernel = labels[:, :, :, -1]
    gt_kernel *= gt_text

    return gt_kernel, kernel


def filter_input(metric_type):
    if metric_type == config.KERNEL_METRICS:
        return filter_kernels
    elif metric_type == config.TEXT_METRICS:
        return filter_texts
    else:
        raise NotImplementedError("The metric type has not been recognised.")


def compute_confusion_matrix(
    labels, predictions, masks, metric_type, n_classes=2
):
    y_true, y_pred = filter_input(metric_type)(labels, predictions, masks)

    def _compute_confusion(ground_truth, prediction):
        mask = tf.logical_and(
            tf.greater_equal(ground_truth, 0), tf.less(ground_truth, n_classes)
        )
        masked_gt = tf.boolean_mask(ground_truth, mask)
        masked_gt = tf.cast(masked_gt, tf.int32)
        masked_pred = tf.boolean_mask(prediction, mask)
        masked_pred = tf.cast(masked_pred, tf.int32)
        confusion = tf.math.confusion_matrix(
            masked_gt, masked_pred, num_classes=n_classes
        )
        confusion = tf.cast(confusion, tf.float32)
        return confusion

    def get_increment(index):
        ground_truth_increment = y_true[index]
        prediction_increment = y_pred[index]
        return _compute_confusion(ground_truth_increment, prediction_increment)

    batch_size = tf.shape(y_true)[0]
    batch_size = tf.cast(batch_size, tf.int64)
    indices = tf.range(batch_size)

    matrix = tf.math.reduce_mean(
        tf.map_fn(get_increment, indices, dtype=tf.float32), axis=0
    )
    return matrix


def overall_accuracy(
    labels, predictions, masks, metric_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    total_sum = tf.math.reduce_sum(confusion_matrix)
    overall_accuracy = tf.math.divide(
        tf.math.reduce_sum(diagonal), (total_sum + epsilon)
    )
    return tf.compat.v1.metrics.mean(
        overall_accuracy, name="{}/{}".format(metric_type, "overall_accuracy")
    )


def precision(labels, predictions, masks, metric_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    precision = diagonal[0] / (col_sum[0] + epsilon)
    return tf.compat.v1.metrics.mean(
        precision, name="{}/{}".format(metric_type, "precision")
    )


def recall(labels, predictions, masks, metric_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    recall = diagonal[0] / (row_sum[0] + epsilon)
    return tf.compat.v1.metrics.mean(
        recall, name="{}/{}".format(metric_type, "recall")
    )


def f1_score(labels, predictions, masks, metric_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    precision = diagonal[0] / (col_sum[0] + epsilon)
    recall = diagonal[0] / (row_sum[0] + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return tf.compat.v1.metrics.mean(
        f1_score, name="{}/{}".format(metric_type, "f1_score")
    )


def mean_accuracy(
    labels, predictions, masks, metric_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    mean_accuracy = tf.math.reduce_mean(diagonal / (row_sum + epsilon))
    return tf.compat.v1.metrics.mean(
        mean_accuracy, name="{}/{}".format(metric_type, "mean_accuracy")
    )


def mean_iou(labels, predictions, masks, metric_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    iou = diagonal / (row_sum + col_sum - diagonal + epsilon)
    mean_iou = tf.math.reduce_mean(iou)
    return tf.compat.v1.metrics.mean(
        mean_iou, name="{}/{}".format(metric_type, "mean_iou")
    )


def frequency_weighted_accuracy(
    labels, predictions, masks, metric_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, metric_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    columns = tf.math.reduce_sum(confusion_matrix, axis=1)
    rows = tf.math.reduce_sum(confusion_matrix, axis=0)
    total_sum = tf.math.reduce_sum(confusion_matrix)
    iou = diagonal / (columns + rows - diagonal + epsilon)
    frequency = columns / (total_sum + epsilon)
    fwaccuracy = tf.math.reduce_sum(
        tf.boolean_mask(frequency, tf.greater(frequency, 0))
        * tf.boolean_mask(iou, tf.greater(frequency, 0))
    )
    return tf.compat.v1.metrics.mean(
        fwaccuracy, name="{}/{}".format(metric_type, "fwaccuracy")
    )


def build_metrics(labels, predictions, masks):
    kernel_metrics_type = config.KERNEL_METRICS
    kernel_overall_accuracy = overall_accuracy(
        labels, predictions, masks, kernel_metrics_type
    )
    kernel_mean_accuracy = mean_accuracy(
        labels, predictions, masks, kernel_metrics_type
    )
    kernel_mean_iou = mean_iou(labels, predictions, masks, kernel_metrics_type)
    kernel_fwaccuracy = frequency_weighted_accuracy(
        labels, predictions, masks, kernel_metrics_type
    )
    kernel_precision = precision(
        labels, predictions, masks, kernel_metrics_type
    )
    kernel_recall = recall(labels, predictions, masks, kernel_metrics_type)
    kernel_f1_score = f1_score(labels, predictions, masks, kernel_metrics_type)

    text_metrics_type = config.TEXT_METRICS
    text_overall_accuracy = overall_accuracy(
        labels, predictions, masks, text_metrics_type
    )
    text_mean_accuracy = mean_accuracy(
        labels, predictions, masks, text_metrics_type
    )
    text_mean_iou = mean_iou(labels, predictions, masks, text_metrics_type)
    text_fwaccuracy = frequency_weighted_accuracy(
        labels, predictions, masks, text_metrics_type
    )
    text_precision = precision(labels, predictions, masks, text_metrics_type)
    text_recall = recall(labels, predictions, masks, text_metrics_type)
    text_f1_score = f1_score(labels, predictions, masks, text_metrics_type)

    comuputed_metrics = {
        "{}/kernel_overall_accuracy".format(
            kernel_metrics_type
        ): kernel_overall_accuracy,
        "{}/kernel_mean_accuracy".format(
            kernel_metrics_type
        ): kernel_mean_accuracy,
        "{}/kernel_mean_iou".format(kernel_metrics_type): kernel_mean_iou,
        "{}/kernel_fwaccuracy".format(kernel_metrics_type): kernel_fwaccuracy,
        "{}/kernel_precision".format(kernel_metrics_type): kernel_precision,
        "{}/kernel_recall".format(kernel_metrics_type): kernel_recall,
        "{}/kernel_f1_score".format(kernel_metrics_type): kernel_f1_score,
        "{}/text_overall_accuracy".format(
            text_metrics_type
        ): text_overall_accuracy,
        "{}/text_mean_accuracy".format(text_metrics_type): text_mean_accuracy,
        "{}/text_mean_iou".format(text_metrics_type): text_mean_iou,
        "{}/text_fwaccuracy".format(text_metrics_type): text_fwaccuracy,
        "{}/text_precision".format(text_metrics_type): text_precision,
        "{}/text_recall".format(text_metrics_type): text_recall,
        "{}/text_f1_score".format(text_metrics_type): text_f1_score,
    }

    return comuputed_metrics
