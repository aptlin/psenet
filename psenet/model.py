from psenet import config
import tensorflow as tf
from psenet.nets.fpn import FPN


def build_model(params):
    images = tf.keras.Input(
        shape=[None, None, 3], name=config.IMAGE, dtype=tf.float32
    )
    kernels = FPN(
        backbone_name=params.backbone_name,
        input_shape=(None, None, 3),
        classes=params.kernel_num,
        activation="linear",
        weights=None,
        encoder_weights=params.encoder_weights,
        encoder_features="default",
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation="concat",
        pyramid_dropout=None,
    )(images)

    logits = tf.keras.Model(
        inputs={config.IMAGE: images},
        outputs={config.KERNELS: kernels},
        name=config.KERNELS,
    )
    return logits
