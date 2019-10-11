import abc

import pymia.deeplearning.model as mdl
import tensorflow as tf

import pc.configuration.config as cfg


class BaseModel(mdl.TensorFlowModel, abc.ABC):

    def placeholders(self, x_shape: tuple, y_shape: tuple):
        self.x_placeholder = tf.placeholder(tf.float32, x_shape, 'images')
        self.y_placeholder = tf.placeholder(tf.int32, y_shape, 'labels')
        self.is_training_placeholder = tf.placeholder(tf.bool,
                                                      name='is_training')  # True if training phase, otherwise False

        cube_shape = y_shape + (self.image_information_config.spatial_size, self.image_information_config.spatial_size,
                                self.image_information_config.spatial_size, self.image_information_config.ch_in)
        self.image_information_placeholder = tf.placeholder(tf.float32, cube_shape, 'image_information')
        self.image_information_label_placeholder = tf.placeholder(tf.int32, cube_shape[:-1], 'image_information_labels')

    def __init__(self, session, sample: dict, config: cfg.Configuration):
        self.no_points = config.no_points
        self.use_point_feature = config.use_point_feature
        self.use_image_information = config.use_image_information

        self.channel_factor = config.channel_factor

        self.learning_rate = config.learning_rate
        self.dropout_p = config.dropout_p
        self.image_information_config = config.image_information_config

        self.image_information_placeholder = None
        self.image_information_label_placeholder = None

        self.current_epoch = 1

        # call base class constructor after initializing variables used by the implemented abstract functions
        super().__init__(session, config.model_dir,
                         x_shape=(None,) + sample['images'].shape,
                         y_shape=(None,) + sample['labels'].shape)

        self.add_summaries()

    def epoch_summaries(self) -> list:
        return []

    def batch_summaries(self):
        return []

    def visualization_summaries(self):
        return [], [], [], []

    def add_summaries(self):
        # tf.summary.scalar('train/learning_rate', self.learning_rate)  # todo
        tf.summary.scalar('train/loss', self.loss)  # todo

    def loss_function(self, prediction, label=None, **kwargs):
        if isinstance(prediction, tuple):
            loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction[0], labels=label)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=label)

        loss = tf.identity(loss, name='loss')
        return loss

    def optimize(self, **kwargs):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # add extra operations of the graph to the optimizer
        # e.g. this is used for batch normalization
        # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.session.run(self.epoch_op, feed_dict={self.epoch_placeholder: epoch})
