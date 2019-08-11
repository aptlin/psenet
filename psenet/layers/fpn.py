import tensorflow as tf
from psenet.layers.common import Conv2dBn
from psenet.backbones.factory import Backbones


def Conv3x3BnReLU(filters, use_batchnorm, name=None, weight_decay=None):
    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            use_batchnorm=use_batchnorm,
            name=name,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            if weight_decay
            else None,
        )(input_tensor)

    return wrapper


def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None, weight_decay=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + "a"
        name2 = name + "b"

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(
            filters, use_batchnorm, name=name1, weight_decay=weight_decay
        )(input_tensor)
        x = Conv3x3BnReLU(
            filters, use_batchnorm, name=name2, weight_decay=weight_decay
        )(x)
        return x

    return wrapper


def FPNBlock(pyramid_filters, stage, weight_decay=None):
    conv0_name = "fpn_stage_p{}_pre_conv".format(stage)
    conv1_name = "fpn_stage_p{}_conv".format(stage)
    add_name = "fpn_stage_p{}_add".format(stage)
    up_name = "fpn_stage_p{}_upsampling".format(stage)

    channels_axis = (
        3 if tf.keras.backend.image_data_format() == "channels_last" else 1
    )

    def wrapper(input_tensor, skip):
        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
        input_filters = tf.keras.backend.int_shape(input_tensor)[channels_axis]
        if input_filters != pyramid_filters:
            input_tensor = tf.keras.layers.Conv2D(
                filters=pyramid_filters,
                kernel_size=(1, 1),
                kernel_initializer="he_uniform",
                name=conv0_name,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
                if weight_decay
                else None,
            )(input_tensor)

        skip = tf.keras.layers.Conv2D(
            filters=pyramid_filters,
            kernel_size=(1, 1),
            kernel_initializer="he_uniform",
            name=conv1_name,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            if weight_decay
            else None,
        )(skip)

        x = tf.keras.layers.UpSampling2D((2, 2), name=up_name)(input_tensor)
        x = tf.keras.layers.Add(name=add_name)([x, skip])

        return x

    return wrapper


# ---------------------------------------------------------------------
#  FPN Decoder
# ---------------------------------------------------------------------


def build_fpn(
    backbone,
    skip_connection_layers,
    pyramid_filters=256,
    segmentation_filters=128,
    classes=1,
    activation="sigmoid",
    use_batchnorm=True,
    aggregation="sum",
    dropout=None,
    weight_decay=None,
):
    inputs = backbone.input
    outputs = backbone.output

    # building decoder blocks with skip connections
    skips = [
        backbone.get_layer(name=i).output
        if isinstance(i, str)
        else backbone.get_layer(index=i).output
        for i in skip_connection_layers
    ]

    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5, weight_decay=weight_decay)(
        outputs, skips[0]
    )
    p4 = FPNBlock(pyramid_filters, stage=4, weight_decay=weight_decay)(
        p5, skips[1]
    )
    p3 = FPNBlock(pyramid_filters, stage=3, weight_decay=weight_decay)(
        p4, skips[2]
    )
    p2 = FPNBlock(pyramid_filters, stage=2, weight_decay=weight_decay)(
        p3, skips[3]
    )

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(
        segmentation_filters,
        use_batchnorm,
        name="segm_stage5",
        weight_decay=weight_decay,
    )(p5)
    s4 = DoubleConv3x3BnReLU(
        segmentation_filters,
        use_batchnorm,
        name="segm_stage4",
        weight_decay=weight_decay,
    )(p4)
    s3 = DoubleConv3x3BnReLU(
        segmentation_filters,
        use_batchnorm,
        name="segm_stage3",
        weight_decay=weight_decay,
    )(p3)
    s2 = DoubleConv3x3BnReLU(
        segmentation_filters,
        use_batchnorm,
        name="segm_stage2",
        weight_decay=weight_decay,
    )(p2)

    # upsampling to same resolution
    s5 = tf.keras.layers.UpSampling2D(
        (8, 8), interpolation="nearest", name="upsampling_stage5"
    )(s5)
    s4 = tf.keras.layers.UpSampling2D(
        (4, 4), interpolation="nearest", name="upsampling_stage4"
    )(s4)
    s3 = tf.keras.layers.UpSampling2D(
        (2, 2), interpolation="nearest", name="upsampling_stage3"
    )(s3)

    # aggregating results
    if aggregation == "sum":
        outputs = tf.keras.layers.Add(name="aggregation_sum")([s2, s3, s4, s5])
    elif aggregation == "concat":
        concat_axis = (
            3 if tf.keras.backend.image_data_format() == "channels_last" else 1
        )
        outputs = tf.keras.layers.Concatenate(
            axis=concat_axis, name="aggregation_concat"
        )([s2, s3, s4, s5])
    else:
        raise ValueError(
            'Aggregation parameter should be in ("sum", "concat"), '
            "got {}".format(aggregation)
        )

    if dropout:
        outputs = tf.keras.layers.SpatialDropout2D(
            dropout, name="pyramid_dropout"
        )(outputs)

    # final stage
    outputs = Conv3x3BnReLU(
        segmentation_filters,
        use_batchnorm,
        name="final_stage",
        weight_decay=weight_decay,
    )(outputs)
    outputs = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="final_upsampling"
    )(outputs)

    # model head (define number of output classes)
    outputs = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        name="head_conv",
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        if weight_decay
        else None,
    )(outputs)
    outputs = tf.keras.layers.Activation(activation, name=activation)(outputs)

    # create keras model instance
    model = tf.keras.Model(inputs, outputs)

    return model


# ---------------------------------------------------------------------
#  FPN Model
# ---------------------------------------------------------------------


def FPN(
    backbone_name="mobilenetv2",
    backbone=None,
    input_shape=(None, None, 3),
    classes=7,
    activation="sigmoid",
    weights=None,
    encoder_weights="imagenet",
    encoder_features="default",
    pyramid_block_filters=256,
    pyramid_use_batchnorm=True,
    pyramid_aggregation="concat",
    pyramid_dropout=None,
    weight_decay=None,
):
    if backbone_name and not backbone:
        backbone = Backbones.get_backbone(
            backbone_name,
            input_shape=input_shape,
            weights=encoder_weights,
            include_top=False,
        )

    if encoder_features == "default":
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_fpn(
        backbone=backbone,
        skip_connection_layers=encoder_features,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=pyramid_block_filters // 2,
        use_batchnorm=pyramid_use_batchnorm,
        dropout=pyramid_dropout,
        activation=activation,
        classes=classes,
        aggregation=pyramid_aggregation,
        weight_decay=weight_decay,
    )

    if weights is not None:
        model.load_weights(weights)

    return model
