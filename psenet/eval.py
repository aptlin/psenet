import tensorflow as tf
from psenet import config


def build_eval_exporter():
    def serving_input_fn():
        features = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            ),
            config.MASK: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None]
            ),
        }
        receiver_tensors = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            ),
            config.MASK: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None]
            ),
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors
        )

    return tf.estimator.LatestExporter(
        name="exporter", serving_input_receiver_fn=serving_input_fn
    )
