import tensorflow as tf

# tf.enable_eager_execution()
from psenet.data.generator import Dataset

dataset = Dataset("./dist/mlt/tfrecords", 10, num_readers=4)
it = dataset.build().make_one_shot_iterator()
s = it.get_next()
with tf.Session() as sess:
    for i in range(100):
        print(sess.run(s))

