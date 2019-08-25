from psenet import config
import tensorflow as tf


def build_optimizer(FLAGS):
    return tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            FLAGS.learning_rate,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
            staircase=True,
        ),
        momentum=config.MOMENTUM,
    )
