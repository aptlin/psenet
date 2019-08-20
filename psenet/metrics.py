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


def filter_input(input_type):
    if input_type == config.KERNEL_METRICS:
        return filter_kernels
    elif input_type == config.TEXT_METRICS:
        return filter_texts
    else:
        raise NotImplementedError("The metric type has not been recognised.")


def compute_confusion_matrix(
    labels, predictions, masks, input_type, n_classes=2
):
    y_true, y_pred = filter_input(input_type)(labels, predictions, masks)

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


class OverallAccuracy(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="overall_accuracy", **kwargs):
        super(OverallAccuracy, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.overall_accuracy_sum = self.add_weight(
            name="overall_accuracy_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="overall_accuracy_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = overall_accuracy(ground_truth, y_pred, masks, self.input_type)
        self.overall_accuracy_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.overall_accuracy_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.overall_accuracy_sum.assign(0.0)
        self.count.assign(0.0)


def overall_accuracy(
    labels, predictions, masks, input_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    total_sum = tf.math.reduce_sum(confusion_matrix)
    overall_accuracy = tf.math.divide(
        tf.math.reduce_sum(diagonal), (total_sum + epsilon)
    )
    return tf.identity(
        overall_accuracy, name="{}/{}".format(input_type, "overall_accuracy")
    )


class Precision(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="precision", **kwargs):
        super(Precision, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.precision_sum = self.add_weight(
            name="precision_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="precision_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = precision(ground_truth, y_pred, masks, self.input_type)
        self.precision_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.precision_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.precision_sum.assign(0.0)
        self.count.assign(0.0)


def precision(labels, predictions, masks, input_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    precision = diagonal[0] / (col_sum[0] + epsilon)
    return tf.identity(precision, name="{}/{}".format(input_type, "precision"))


class Recall(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="recall", **kwargs):
        super(Recall, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.recall_sum = self.add_weight(
            name="recall_sum", initializer="zeros"
        )
        self.count = self.add_weight(name="recall_count", initializer="zeros")
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = recall(ground_truth, y_pred, masks, self.input_type)
        self.recall_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.recall_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.recall_sum.assign(0.0)
        self.count.assign(0.0)


def recall(labels, predictions, masks, input_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    recall = diagonal[0] / (row_sum[0] + epsilon)
    return tf.identity(recall, name="{}/{}".format(input_type, "recall"))


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="f1_score", **kwargs):
        super(F1Score, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.f1_score_sum = self.add_weight(
            name="f1_score_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="f1_score_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = f1_score(ground_truth, y_pred, masks, self.input_type)
        self.f1_score_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.f1_score_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.f1_score_sum.assign(0.0)
        self.count.assign(0.0)


def f1_score(labels, predictions, masks, input_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    precision = diagonal[0] / (col_sum[0] + epsilon)
    recall = diagonal[0] / (row_sum[0] + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return tf.identity(f1_score, name="{}/{}".format(input_type, "f1_score"))


class MeanAccuracy(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="mean_accuracy", **kwargs):
        super(MeanAccuracy, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.mean_accuracy_sum = self.add_weight(
            name="mean_accuracy_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="mean_accuracy_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = mean_accuracy(ground_truth, y_pred, masks, self.input_type)
        self.mean_accuracy_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.mean_accuracy_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mean_accuracy_sum.assign(0.0)
        self.count.assign(0.0)


def mean_accuracy(
    labels, predictions, masks, input_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    mean_accuracy = tf.math.reduce_mean(diagonal / (row_sum + epsilon))
    return tf.identity(
        mean_accuracy, name="{}/{}".format(input_type, "mean_accuracy")
    )


class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, input_type, name="mean_iou", **kwargs):
        super(MeanIoU, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.mean_iou_sum = self.add_weight(
            name="mean_iou_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="mean_iou_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = mean_iou(ground_truth, y_pred, masks, self.input_type)
        self.mean_iou_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.mean_iou_sum, self.count)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mean_iou_sum.assign(0.0)
        self.count.assign(0.0)


def mean_iou(labels, predictions, masks, input_type, epsilon=config.EPSILON):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
    )
    diagonal = tf.linalg.diag_part(confusion_matrix)
    row_sum = tf.math.reduce_sum(confusion_matrix, axis=1)
    col_sum = tf.math.reduce_sum(confusion_matrix, axis=0)
    iou = diagonal / (row_sum + col_sum - diagonal + epsilon)
    mean_iou = tf.math.reduce_mean(iou)
    return tf.identity(mean_iou, name="{}/{}".format(input_type, "mean_iou"))


class FrequencyWeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self, input_type, name="frequency_weighted_accuracy", **kwargs
    ):
        super(FrequencyWeightedAccuracy, self).__init__(
            name="{}/{}".format(input_type, name), **kwargs
        )
        self.frequency_weighted_accuracy_sum = self.add_weight(
            name="frequency_weighted_accuracy_sum", initializer="zeros"
        )
        self.count = self.add_weight(
            name="frequency_weighted_accuracy_count", initializer="zeros"
        )
        self.input_type = input_type

    def update_state(self, y_true, y_pred, sample_weight=None):
        masks = y_true[:, :, :, 0]
        ground_truth = y_true[:, :, :, 1:]
        value = frequency_weighted_accuracy(
            ground_truth, y_pred, masks, self.input_type
        )
        self.frequency_weighted_accuracy_sum.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(
            self.frequency_weighted_accuracy_sum, self.count
        )

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.frequency_weighted_accuracy_sum.assign(0.0)
        self.count.assign(0.0)


def frequency_weighted_accuracy(
    labels, predictions, masks, input_type, epsilon=config.EPSILON
):
    confusion_matrix = compute_confusion_matrix(
        labels, predictions, masks, input_type
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
    return tf.identity(
        fwaccuracy, name="{}/{}".format(input_type, "fwaccuracy")
    )


def keras_psenet_metrics():
    ingots = [
        OverallAccuracy,
        MeanAccuracy,
        MeanIoU,
        FrequencyWeightedAccuracy,
        Precision,
        Recall,
        F1Score,
    ]
    kernel_metrics_type = config.KERNEL_METRICS
    kernel_metrics = [Metric(kernel_metrics_type) for Metric in ingots]
    text_metrics_type = config.TEXT_METRICS
    text_metrics = [Metric(text_metrics_type) for Metric in ingots]
    return [*kernel_metrics, *text_metrics]
    # kernel_metrics_type = config.KERNEL_METRICS
    # kernel_overall_accuracy = OverallAccuracy(kernel_metrics_type)
    # kernel_mean_accuracy = MeanAccuracy(kernel_metrics_type)
    # kernel_mean_iou = MeanIoU(kernel_metrics_type)
    # kernel_fwaccuracy = FrequencyWeightedAccuracy(kernel_metrics_type)
    # kernel_precision = Precision(kernel_metrics_type)
    # kernel_recall = Recall(kernel_metrics_type)
    # kernel_f1_score = F1Score(kernel_metrics_type)
    # text_metrics_type = config.TEXT_METRICS
    # text_overall_accuracy = OverallAccuracy(text_metrics_type)
    # text_mean_accuracy = MeanAccuracy(text_metrics_type)
    # text_mean_iou = MeanIoU(text_metrics_type)
    # text_fwaccuracy = FrequencyWeightedAccuracy(text_metrics_type)
    # text_precision = Precision(text_metrics_type)
    # text_recall = Recall(text_metrics_type)
    # text_f1_score = F1Score(text_metrics_type)
    # return [
    #     kernel_overall_accuracy,
    #     kernel_mean_accuracy,
    #     kernel_mean_iou,
    #     kernel_fwaccuracy,
    #     kernel_precision,
    #     kernel_recall,
    #     kernel_f1_score,
    #     text_overall_accuracy,
    #     text_mean_accuracy,
    #     text_mean_iou,
    #     text_fwaccuracy,
    #     text_precision,
    #     text_recall,
    #     text_f1_score,
    # ]


def psenet_metrics(labels, predictions):
    masks = labels[:, :, :, 0]
    ground_truth = labels[:, :, :, 1:]
    kernel_metrics_type = config.KERNEL_METRICS
    kernel_overall_accuracy = overall_accuracy(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_mean_accuracy = mean_accuracy(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_mean_iou = mean_iou(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_fwaccuracy = frequency_weighted_accuracy(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_precision = precision(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_recall = recall(
        ground_truth, predictions, masks, kernel_metrics_type
    )
    kernel_f1_score = f1_score(
        ground_truth, predictions, masks, kernel_metrics_type
    )

    text_metrics_type = config.TEXT_METRICS
    text_overall_accuracy = overall_accuracy(
        ground_truth, predictions, masks, text_metrics_type
    )
    text_mean_accuracy = mean_accuracy(
        ground_truth, predictions, masks, text_metrics_type
    )
    text_mean_iou = mean_iou(
        ground_truth, predictions, masks, text_metrics_type
    )
    text_fwaccuracy = frequency_weighted_accuracy(
        ground_truth, predictions, masks, text_metrics_type
    )
    text_precision = precision(
        ground_truth, predictions, masks, text_metrics_type
    )
    text_recall = recall(ground_truth, predictions, masks, text_metrics_type)
    text_f1_score = f1_score(
        ground_truth, predictions, masks, text_metrics_type
    )

    computed_metrics = {
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

    return computed_metrics
