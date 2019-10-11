import os
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

sampling_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))

farthest_point_sample = sampling_module.farthest_point_sample
