import tensorflow as tf
from psenet import config


def filter_texts(labels, predictions, kernel_num):
    gt_texts = labels[:, :, :, 0]
    masks = labels[:, :, :, kernel_num]
    texts = predictions[:, :, :, 0]
    texts = tf.math.sigmoid(texts) * masks
    texts = tf.where(
        tf.greater(texts, 0.5), tf.ones_like(texts), tf.zeros_like(texts)
    )
    gt_text = gt_texts * masks
    return gt_text, texts


def filter_kernels(labels, predictions, kernel_num):
    gt_texts = labels[:, :, :, 0]
    gt_kernels = labels[:, :, :, 1:kernel_num]
    masks = labels[:, :, :, kernel_num]
    mask = gt_texts * masks

    kernel = predictions[:, :, :, -1]
    kernel = tf.math.sigmoid(kernel)
    kernel = tf.where(
        tf.greater(kernel, 0.5), tf.ones_like(kernel), tf.zeros_like(kernel)
    )
    kernel *= mask
    kernel = tf.cast(kernel, tf.float32)

    gt_kernel = gt_kernels[:, :, :, -1]
    gt_kernel *= mask
    return gt_kernel, kernel


def confusion_matrix(y_true, y_pred, n_classes):
    def _compute_confusion(ground_truth, prediction):
        mask = tf.logical_and(
            tf.greater_equal(ground_truth, 0), tf.less(ground_truth, n_classes)
        )
        masked_gt = tf.boolean_mask(ground_truth, mask)
        masked_gt = tf.cast(masked_gt, tf.int32)
        masked_pred = tf.boolean_mask(prediction, mask)
        masked_pred = tf.cast(masked_gt, tf.int32)
        confusion = tf.reshape(
            tf.math.bincount(
                masked_gt * n_classes + masked_pred, minlength=n_classes ** 2
            ),
            [n_classes, n_classes],
        )
        confusion = tf.cast(confusion, tf.float32)
        return confusion

    def get_increment(index):
        ground_truth_increment = y_true[index]
        ground_truth_increment = tf.reshape(ground_truth_increment, [-1])
        prediction_increment = y_pred[index]
        prediction_increment = tf.reshape(prediction_increment, [-1])
        return _compute_confusion(ground_truth_increment, prediction_increment)

    n_labels = tf.shape(y_true)[0]
    n_labels = tf.cast(n_labels, tf.int64)
    indices = tf.range(n_labels)

    matrix = tf.math.reduce_sum(
        tf.map_fn(get_increment, indices, dtype=tf.float32), axis=0
    )

    return matrix


def overall_accuracy(confusion_matrix):
    epsilon = config.EPSILON
    diagonal = tf.linalg.diag_part(confusion_matrix)
    total_sum = tf.math.reduce_sum(confusion_matrix)
    overall_accuracy = tf.math.divide(
        tf.math.reduce_sum(diagonal), (total_sum + epsilon)
    )
    return overall_accuracy


def mean_accuracy(confusion_matrix):
    epsilon = config.EPSILON
    diagonal = tf.linalg.diag_part(confusion_matrix)
    columns = tf.math.reduce_sum(confusion_matrix, axis=1)
    mean_accuracy = tf.math.reduce_mean(diagonal / (columns + epsilon))
    return mean_accuracy


def mean_iou(confusion_matrix):
    epsilon = config.EPSILON
    diagonal = tf.linalg.diag_part(confusion_matrix)
    columns = tf.math.reduce_sum(confusion_matrix, axis=1)
    rows = tf.math.reduce_sum(confusion_matrix, axis=0)
    iou = diagonal / (columns + rows - diagonal + epsilon)
    mean_iou = tf.math.reduce_mean(iou)
    return mean_iou


def frequency_weighted_accuracy(confusion_matrix):
    epsilon = config.EPSILON
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
    return fwaccuracy


def filter_input(metric_type):
    if metric_type == config.KERNEL_METRICS:
        return filter_kernels
    elif metric_type == config.TEXT_METRICS:
        return filter_texts
    else:
        raise NotImplementedError("The metric type has not been recognised.")


def build_metrics(
    kernel_num,
    confusion_matrix_metrics={
        "overall_accuracy": overall_accuracy,
        "mean_accuracy": mean_accuracy,
        "mean_iou": mean_iou,
        "frequency_weighted_accuracy": frequency_weighted_accuracy,
    },
    metric_names=[config.TEXT_METRICS, config.KERNEL_METRICS],
):
    def compute(labels, predictions):
        computed_metrics = {}
        for name in metric_names:
            y_true, y_pred = filter_input(name)(
                labels, predictions[config.KERNELS], kernel_num
            )
            matrix = confusion_matrix(y_true, y_pred, 2)
            confusion_label = f"{name}/confusion-matrix"
            mean_confusion, upd_confusion = tf.compat.v1.metrics.mean_tensor(
                matrix, name=confusion_label
            )
            computed_metrics[confusion_label] = (mean_confusion, upd_confusion)
            print_confusion = tf.print("before upd:", matrix)
            print_mean_confusion = tf.print("after upd:", mean_confusion)
            with tf.control_dependencies(
                [print_confusion, print_mean_confusion]
            ):
                for metric_type, metric in confusion_matrix_metrics.items():
                    val = metric(mean_confusion)
                    label = f"{name}/{metric_type}"
                    computed_metrics[label] = tf.compat.v1.metrics.mean(
                        val, name=label
                    )
        return computed_metrics

    return compute
