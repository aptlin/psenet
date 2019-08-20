import tensorflow as tf
from psenet import config


def build_eval_exporter():
    def serving_input_fn():
        features = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            )
        }
        receiver_tensors = {
            config.IMAGE: tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3]
            )
        }
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors
        )

    return tf.estimator.LatestExporter(
        name="exporter", serving_input_receiver_fn=serving_input_fn
    )
