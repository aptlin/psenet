import tensorflow as tf

tf.enable_eager_execution()
from psenet.data.generator import Dataset

dataset = Dataset("./dist/mlt/tfrecords", 100)
it = dataset.get_one_shot_iterator()
s = it.get_next()
# sess = tf.InteractiveSession() 


