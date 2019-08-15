# NOTICE: This code is based on the work by Pavel Yakubovskiy.
# See https://github.com/qubvel/segmentation_models for details.
#
# The MIT License
#
# Copyright (c) 2018, Pavel Yakubovskiy, Sasha Illarionov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import tensorflow.keras.applications as ka


class ModelsFactory:
    _models = {
        # Resnet V2
        "resnet50v2": [ka.resnet50.ResNet50, ka.resnet50.preprocess_input],
        # VGG
        "vgg16": [ka.vgg16.VGG16, ka.vgg16.preprocess_input],
        "vgg19": [ka.vgg19.VGG19, ka.vgg19.preprocess_input],
        # Densenet
        "densenet121": [ka.densenet.DenseNet121, ka.densenet.preprocess_input],
        "densenet169": [ka.densenet.DenseNet169, ka.densenet.preprocess_input],
        "densenet201": [ka.densenet.DenseNet201, ka.densenet.preprocess_input],
        # Inception
        "inceptionresnetv2": [
            ka.inception_resnet_v2.InceptionResNetV2,
            ka.inception_resnet_v2.preprocess_input,
        ],
        "inceptionv3": [
            ka.inception_v3.InceptionV3,
            ka.inception_v3.preprocess_input,
        ],
        "xception": [ka.xception.Xception, ka.xception.preprocess_input],
        # Nasnet
        "nasnetlarge": [ka.nasnet.NASNetLarge, ka.nasnet.preprocess_input],
        "nasnetmobile": [ka.nasnet.NASNetMobile, ka.nasnet.preprocess_input],
        # MobileNet
        "mobilenet": [ka.mobilenet.MobileNet, ka.mobilenet.preprocess_input],
        "mobilenetv2": [
            ka.mobilenet_v2.MobileNetV2,
            ka.mobilenet_v2.preprocess_input,
        ],
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    def get(self, name):
        if name not in self.models_names():
            raise ValueError(
                "No such model `{}`, available models: {}".format(
                    name, list(self.models_names())
                )
            )

        model_fn, preprocess_input = self.models[name]
        return model_fn, preprocess_input


class BackbonesFactory(ModelsFactory):
    def __init__(self):
        self._default_feature_layers = {
            # VGG
            "vgg16": (
                "block5_conv3",
                "block4_conv3",
                "block3_conv3",
                "block2_conv2",
                "block1_conv2",
            ),
            "vgg19": (
                "block5_conv4",
                "block4_conv4",
                "block3_conv4",
                "block2_conv2",
                "block1_conv2",
            ),
            # ResNets
            "resnet18": (
                "stage4_unit1_relu1",
                "stage3_unit1_relu1",
                "stage2_unit1_relu1",
                "relu0",
            ),
            "resnet34": (
                "stage4_unit1_relu1",
                "stage3_unit1_relu1",
                "stage2_unit1_relu1",
                "relu0",
            ),
            "resnet50": (
                "stage4_unit1_relu1",
                "stage3_unit1_relu1",
                "stage2_unit1_relu1",
                "relu0",
            ),
            "resnet101": (
                "stage4_unit1_relu1",
                "stage3_unit1_relu1",
                "stage2_unit1_relu1",
                "relu0",
            ),
            "resnet152": (
                "stage4_unit1_relu1",
                "stage3_unit1_relu1",
                "stage2_unit1_relu1",
                "relu0",
            ),
            # Inception
            "inceptionv3": (228, 86, 16, 9),
            "inceptionresnetv2": (594, 260, 16, 9),
            # DenseNet
            "densenet121": (311, 139, 51, 4),
            "densenet169": (367, 139, 51, 4),
            "densenet201": (479, 139, 51, 4),
            # Mobile Nets
            "mobilenet": (
                "conv_pw_11_relu",
                "conv_pw_5_relu",
                "conv_pw_3_relu",
                "conv_pw_1_relu",
            ),
            "mobilenetv2": (
                "block_13_expand_relu",
                "block_6_expand_relu",
                "block_3_expand_relu",
                "block_1_expand_relu",
            ),
        }

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]


Backbones = BackbonesFactory()
