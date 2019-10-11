import numpy as np
import tensorflow as tf

import pc.configuration.config as cfg


class Base:

    def __init__(self, config: cfg.ImageInformationConfiguration, no_points: int, is_training_placeholder):
        self.ch_in = config.ch_in
        self.spatial_size = config.spatial_size
        self.no_points = no_points
        self.no_features_latent_space = config.no_features
        # todo: add calculation of no_features_latent_space based on spatial_size and config.no_features
        # --> for CS =5, NO_FEAT=64 --> no_feat_latent=8

        self.norm = 'bn'
        self.is_training_placeholder = is_training_placeholder
        self.layer_no = 1

    def conv3d(self, x, filters: int, kernel_size: int, padding: str = 'valid',
               dropout_p: float = 0.2, norm: str = 'bn', activation=tf.nn.relu):
        x = tf.layers.conv3d(x, filters=filters, kernel_size=kernel_size, padding=padding,
                             activation=activation, name='layer{}_conv'.format(self.layer_no))
        self.layer_no += 1

        if dropout_p > 0:
            x = tf.layers.dropout(x, dropout_p, training=self.is_training_placeholder)

        if norm == 'bn':
            x = tf.layers.batch_normalization(x, training=self.is_training_placeholder)

        return x

    def down_conv(self, x, no_channels):
        x = self.conv3d(x, no_channels, 3, 'same', 0.0, self.norm, tf.nn.relu)
        x = self.conv3d(x, no_channels, 3, 'same', 0.0, self.norm, tf.nn.relu)
        x = tf.layers.max_pooling3d(x, 2, 2, 'same')
        return x


class Encoder(Base):

    def __init__(self, config: cfg.ImageInformationConfiguration, no_points: int, dropout_p, is_training_placeholder):
        super().__init__(config, no_points, is_training_placeholder)
        self.dropout_p = dropout_p
        self.no_features_intermediate = self.no_features_latent_space // 2

    def inference(self, x):
        x = tf.reshape(x, (-1,) + tuple(x.shape[2:].as_list()))  # shape is (B * N, X, Y, C)

        x = self.down_conv(x, self.no_features_intermediate)
        x = self.down_conv(x, self.no_features_latent_space)

        x = tf.reshape(x, (-1, np.prod(x.shape[1:])))
        x = tf.reshape(x, (-1, self.no_points, x.shape[-1]))
        x = tf.layers.dropout(x, self.dropout_p, training=self.is_training_placeholder)
        return x
