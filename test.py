import tensorflow as tf

# tf.enable_eager_execution()
from psenet.data.generator import Dataset

dataset = Dataset("./dist/mlt/tfrecords", 100, num_readers=4)
it = dataset.build().make_one_shot_iterator()
s = it.get_next()
with tf.Session() as sess:
    for i in range(100):
        print(sess.run(s))

# import tensorflow as tf
# from segmentation_models import FPN


# def build_fpn(backbone_name: str, n_kernels=7):
#     return FPN(
#         backbone_name=backbone_name,
#         classes=n_kernels,
#         encoder_weights="imagenet",
#         activation="sigmoid",
#         pyramid_block_filters=256,
#     )
# fpn = build_fpn('mobilenetv2')
# x = tf.random.normal([1, 6, 6, 3])
# fpn(x)