from segmentation_models import FPN


def feature_pyramid_network(self, params):
    return FPN(
        backbone_name=params.backbone_name,
        input_shape=(None, None, 3),
        classes=params.n_kernels,
        encoder_weights="imagenet",
        encoder_freeze=False,
        activation="sigmoid",
        final_interpolation="nearest",
        pyramid_block_filters=256,
    )
