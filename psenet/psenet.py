import tensorflow as tf
from tensorflow.keras import Model
from segmentation_models import FPN


class PSENet(Model):
    """ Attention-based Sequence Recognition Network
    """

    def __init__(self, backbone_name: str, n_kernels=7):
        super(PSENet, self).__init__()
        self.fpn = FPN(
            backbone_name=backbone_name,
            classes=n_kernels,
            encoder_weights="imagenet",
            activation="sigmoid",
            pyramid_block_filters=256,
        )

    def call(self, inputs, training=False):
        inputs_shape = tf.shape(inputs)
        assert inputs_shape[1] % 32 == 0 and inputs_shape[2] % 32 == 0
        return self.fpn(inputs)

