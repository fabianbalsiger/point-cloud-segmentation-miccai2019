"""Code adopted from https://github.com/yangyanli/PointCNN"""
import math

import tensorflow as tf

import pc.configuration.config as cfg
import pc.model.base as base
import pc.model.point_cnn_patch as patch
import pc.model.point_cnn_util as pf
import pc.model.sampling.sampling as sampling


def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, depth_multiplier,
          with_global=False):
    """

    Args:
        pts: The input point cloud of size (B, N, DIM), e.g., (None, 2048, 3).
        fts: The features of the input point cloud of size (B, N, C1), e.g., (None, 2048, 1)
        qrs: The representative points of size (B, N2, DIM), e.g., (None, 768, 3)
        tag:
        N: The batch size.
        K: Number of neighbors to convolve over?
        D: Dilation rate
        P: Number of representative points
        C: The feature dimensionality.
        C_pts_fts:
        is_training:
        depth_multiplier:
        with_global:

    Returns:

    """
    # dilate point cloud
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]

    # move P to local coordinate system of p (line 1 in Algorithm 1)
    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed (line 2 in Algorithm 1)
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)

    # concatenate features (line 3 in Algorithm 1)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    # X-transformation (line 4 in Algorithm 1)
    X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
    X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
    X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
    X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
    X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
    X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')

    # (line 5 in Algorithm 1)
    fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')

    # (line 6 in Algorithm 1)
    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN(base.BaseModel):

    def inference(self, x) -> object:

        # x has shape (B, N, C)
        if x.shape[-1] > 3:
            input_points = x[..., 0:3]
            if self.use_point_feature:
                point_features = x[..., 3:]
            else:
                point_features = None
        else:
            input_points = x
            point_features = None

        # extract image information features
        if self.use_image_information:
            e = patch.Encoder(self.image_information_config, self.no_points, self.dropout_p, self.is_training_placeholder)
            image_information_features = e.inference(self.image_information_placeholder)

            if point_features is not None:
                point_features = tf.concat([point_features, image_information_features], axis=-1)
            else:
                point_features = image_information_features

        xconv_param_name = ('K', 'D', 'P', 'C')
        xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                        [(8, 1, -1, 32 * self.channel_factor),
                         (12, 2, 768, 32 * self.channel_factor),
                         (16, 2, 384, 64 * self.channel_factor),
                         (16, 6, 128, 128 * self.channel_factor)]]

        xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
        xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                         [(16, 6, 3, 2),
                          (12, 6, 2, 1),
                          (8, 6, 1, 0),
                          (8, 4, 0, 0)]]

        fc_param_name = ('C', 'dropout_rate')
        fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                     [(32 * self.channel_factor, self.dropout_p),
                      (32 * self.channel_factor, self.dropout_p)]]

        with_global_setting = True
        batch_size = tf.shape(input_points)[0]

        sampling_config = 'fps'

        list_points = [input_points]
        list_features = [None] if point_features is None else \
            [pf.dense(point_features, xconv_params[0]['C'] // 2, 'features_hd', self.is_training_placeholder)]

        # encoding path
        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']

            # get k-nearest points
            pts = list_points[-1]
            fts = list_features[-1]
            # check if we need to downsample point cloud
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                qrs = list_points[-1]
            else:
                if sampling_config == 'fps':
                    fps_indices = sampling.farthest_point_sample(pts, P)
                    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices, -1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name=tag + 'qrs')  # (N, P, 3)
                elif sampling_config == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                else:
                    raise ValueError('Unknown sampling method "{}"'.format(sampling_config))
            list_points.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (with_global_setting and layer_idx == len(xconv_params) - 1)
            print('ENC', pts.shape, qrs.shape[1], C, K, D)
            fts_xconv = xconv(pts, fts, qrs, tag, batch_size, K, D, P, C, C_pts_fts, self.is_training_placeholder,
                              depth_multiplier, with_global)

            list_features.append(fts_xconv)

        # decoding path
        for layer_idx, layer_param in enumerate(xdconv_params):
            tag = 'xdconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            pts_layer_idx = layer_param['pts_layer_idx']
            qrs_layer_idx = layer_param['qrs_layer_idx']

            pts = list_points[pts_layer_idx + 1]
            fts = list_features[pts_layer_idx + 1] if layer_idx == 0 else list_features[-1]  # fts_fuse is used here
            qrs = list_points[qrs_layer_idx + 1]
            fts_qrs = list_features[qrs_layer_idx + 1]
            P = xconv_params[qrs_layer_idx]['P']
            C = xconv_params[qrs_layer_idx]['C']
            C_prev = xconv_params[pts_layer_idx]['C']
            C_pts_fts = C_prev // 4
            depth_multiplier = 1
            print('DEC', pts.shape, qrs.shape[1], C, K, D)
            fts_xdconv = xconv(pts, fts, qrs, tag, batch_size, K, D, P, C, C_pts_fts, self.is_training_placeholder,
                               depth_multiplier)
            fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
            fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', self.is_training_placeholder)
            list_points.append(qrs)
            list_features.append(fts_fuse)

        # fully-connected at end
        fc_layers = [list_features[-1]]
        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(fc_layers[-1], C, 'fc{:d}'.format(layer_idx), self.is_training_placeholder)
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=self.is_training_placeholder, name='fc{:d}_drop'.format(layer_idx))
            fc_layers.append(fc_drop)

        logits = pf.dense(fc_layers[-1], cfg.NO_CLASSES, 'logits',
                          self.is_training_placeholder, with_bn=False, activation=None)
        return tf.identity(logits, name='network')
