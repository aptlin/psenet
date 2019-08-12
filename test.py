import tensorflow as tf

# tf.enable_eager_execution()
from psenet.data import Dataset
from psenet import config

dataset = Dataset(
    config.TRAINING_DATA_DIR, 1, num_readers=4, should_augment=True
)
it = dataset.build().make_one_shot_iterator()
s = it.get_next()
with tf.Session() as sess:
    d = sess.run(s)
    # for i in range(100):
    #     print(sess.run(s))

# import tensorflow as tf
# from segmentation_models import FPN


# def build_fpn(backbone_name: str, kernel_num=7):
#     return FPN(
#         backbone_name=backbone_name,
#         classes=kernel_num,
#         encoder_weights="imagenet",
#         activation="sigmoid",
#         pyramid_block_filters=256,
#     )
# fpn = build_fpn('mobilenetv2')
# x = tf.random.normal([1, 6, 6, 3])
# fpn(x)
