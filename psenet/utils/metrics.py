from abc import abstractmethod, ABCMeta
import tensorflow as tf

from psenet import config


def filter_texts(labels, predictions, n_kernels):
    gt_texts = labels[:, :, :, 0]
    masks = labels[:, :, :, n_kernels]
    texts = predictions[:, :, :, 0]
    texts = tf.math.sigmoid(texts) * masks
    texts = tf.where(
        tf.greater(texts, 0.5), tf.ones_like(texts), tf.zeros_like(texts)
    )
    gt_text = gt_texts * masks
    return gt_text, texts


def filter_kernels(labels, predictions, n_kernels):
    gt_texts = labels[:, :, :, 0]
    gt_kernels = labels[:, :, :, 1:n_kernels]
    masks = labels[:, :, :, n_kernels]
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


class RunningScore:
    __metaclass__ = ABCMeta

    def __init__(self, name, metric_type, n_classes):
        self.name = name
        self.metric_type = metric_type
        self.n_classes = n_classes
        self.confusion_matrix = tf.zeros(
            [self.n_classes, self.n_classes],
            dtype=tf.float32,
            name=f"{self.name}/{self.metric_type}-confusion-matrix",
        )
        self.label = f"{self.name}/{self.metric_type}-score"

    # @property
    # def confusion_matrix(self):
    #     with tf.variable_scope(
    #         f"{self.name}-{self.metric_type}", reuse=tf.AUTO_REUSE
    #     ):
    #         confusion_matrix = tf.get_variable(
    #             name="confusion-matrix",
    #             initializer=tf.zeros(
    #                 [self.n_classes, self.n_classes],
    #                 dtype=tf.float32,
    #                 name=f"{self.name}/{self.metric_type}-confusion-matrix",
    #             ),
    #             aggregation=tf.VariableAggregation.SUM,
    #         )
    #     return confusion_matrix

    def _compute_confusion(self, ground_truth, prediction):
        mask = tf.logical_and(
            tf.greater_equal(ground_truth, 0),
            tf.less(ground_truth, self.n_classes),
        )
        masked_gt = tf.boolean_mask(ground_truth, mask)
        masked_gt = tf.cast(masked_gt, tf.int32)
        masked_pred = tf.boolean_mask(prediction, mask)
        masked_pred = tf.cast(masked_gt, tf.int32)
        confusion = tf.reshape(
            tf.math.bincount(
                masked_gt * self.n_classes + masked_pred,
                minlength=self.n_classes ** 2,
            ),
            [self.n_classes, self.n_classes],
        )
        confusion = tf.cast(confusion, tf.float32)
        return confusion

    def _update(self, ground_truth, prediction):
        n_labels = tf.shape(ground_truth)[0]
        n_labels = tf.cast(n_labels, tf.int64)
        indices = tf.range(n_labels)

        def get_increment(index):
            ground_truth_increment = ground_truth[index]
            ground_truth_increment = tf.reshape(ground_truth_increment, [-1])
            prediction_increment = prediction[index]
            prediction_increment = tf.reshape(prediction_increment, [-1])
            return self._compute_confusion(
                ground_truth_increment, prediction_increment
            )

        increment = tf.math.reduce_sum(
            tf.map_fn(get_increment, indices, dtype=tf.float32)
        )
        self.confusion_matrix += increment
        # self.confusion_matrix.assign_add(
        #     tf.fill(tf.shape(self.confusion_matrix), increment),
        #     use_locking=True,
        # )

    @abstractmethod
    def __call__(self, ground_truth, prediction):
        return


class OverallAccuracy(RunningScore):
    def __init__(self, name):
        super().__init__(
            n_classes=2, name=name, metric_type="overall-accuracy"
        )

    def __call__(self, ground_truth, prediction):
        self._update(ground_truth, prediction)
        epsilon = config.EPSILON
        diagonal = tf.linalg.diag_part(self.confusion_matrix)
        total_sum = tf.math.reduce_sum(self.confusion_matrix)
        overall_accuracy = tf.math.divide(
            tf.math.reduce_sum(diagonal),
            (total_sum + epsilon),
            name=self.label,
        )
        return overall_accuracy


class MeanAccuracy(RunningScore):
    def __init__(self, name):
        super().__init__(n_classes=2, name=name, metric_type="mean-accuracy")

    def __call__(self, ground_truth, prediction):
        self._update(ground_truth, prediction)
        epsilon = config.EPSILON
        diagonal = tf.linalg.diag_part(self.confusion_matrix)
        columns = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        mean_accuracy = tf.math.reduce_mean(
            diagonal / (columns + epsilon), name=self.label
        )
        return mean_accuracy


class MeanIOU(RunningScore):
    def __init__(self, name):
        super().__init__(
            n_classes=2, name=name, metric_type="mean-intersection-over-union"
        )

    def __call__(self, ground_truth, prediction):
        self._update(ground_truth, prediction)
        epsilon = config.EPSILON
        diagonal = tf.linalg.diag_part(self.confusion_matrix)
        columns = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        rows = tf.math.reduce_sum(self.confusion_matrix, axis=0)
        iou = diagonal / (columns + rows - diagonal + epsilon)
        mean_iou = tf.math.reduce_mean(iou, name=self.label)
        return mean_iou


class FrequencyWeightedAccuracy(RunningScore):
    def __init__(self, name):
        super().__init__(
            n_classes=2, name=name, metric_type="frequency-weighted-accuracy"
        )

    def __call__(self, ground_truth, prediction):
        self._update(ground_truth, prediction)
        epsilon = config.EPSILON
        diagonal = tf.linalg.diag_part(self.confusion_matrix)
        columns = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        rows = tf.math.reduce_sum(self.confusion_matrix, axis=0)
        total_sum = tf.math.reduce_sum(self.confusion_matrix)
        iou = diagonal / (columns + rows - diagonal + epsilon)
        frequency = columns / (total_sum + epsilon)
        fwaccuracy = tf.math.reduce_sum(
            tf.boolean_mask(frequency, tf.greater(frequency, 0))
            * tf.boolean_mask(iou, tf.greater(frequency, 0)),
            name=self.label,
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
    n_kernels,
    metrics=[
        OverallAccuracy,
        MeanAccuracy,
        MeanIOU,
        FrequencyWeightedAccuracy,
    ],
    metric_names=[config.TEXT_METRICS, config.KERNEL_METRICS],
):
    def compute(labels, predictions):
        computed_metrics = {}
        for name in metric_names:
            for metric in metrics:
                y_true, y_pred = filter_input(name)(
                    labels, predictions[config.KERNELS], n_kernels
                )
                metric_object = metric(name)
                val = metric(name)(y_true, y_pred)
                label = metric_object.label
                computed_metrics[label] = tf.metrics.mean(val, name=label)
        return computed_metrics

    return compute
