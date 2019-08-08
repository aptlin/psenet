from segmentation_models import FPN


def feature_pyramid_network(params):
    return FPN(
        backbone_name=params.backbone_name,
        input_shape=(None, None, 3),
        classes=params.kernel_num,
        encoder_weights="imagenet",
        encoder_freeze=False,
        activation="sigmoid",
        pyramid_block_filters=256,
    )
