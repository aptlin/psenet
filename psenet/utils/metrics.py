import tensorflow as tf

from psenet import config


class RunningScore:
    def __init__(self, n_classes, name=""):
        self.name = name
        self.n_classes = n_classes
        self.confusion_matrix = tf.zeros(
            [n_classes, n_classes], dtype=tf.float32
        )

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

    def update(self, ground_truth, prediction):
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

    def compute_scores(self):
        epsilon = config.EPSILON
        diagonal = tf.diag_part(self.confusion_matrix)
        columns = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        rows = tf.math.reduce_sum(self.confusion_matrix, axis=0)
        total_sum = tf.math.reduce_sum(self.confusion_matrix)
        overall_accuracy = tf.math.reduce_sum(diagonal) / (total_sum + epsilon)
        mean_accuracy = tf.math.reduce_mean(diagonal / (columns + epsilon))
        iou = diagonal / (columns + rows - diagonal + epsilon)
        frequency = columns / (total_sum + epsilon)
        fwaccuracy = tf.math.reduce_sum(
            tf.boolean_mask(frequency, tf.greater(frequency, 0))
            * tf.boolean_mask(iou, tf.greater(frequency, 0))
        )

        mean_iou = tf.math.reduce_mean(iou)
        return {
            f"{self.name} :: Overall Accuracy": overall_accuracy,
            f"{self.name} :: Mean Accuracy": mean_accuracy,
            f"{self.name} :: Mean Intersection-over-Union": mean_iou,
            f"{self.name} :: Frequency-Weighted Accuracy": fwaccuracy,
        }
