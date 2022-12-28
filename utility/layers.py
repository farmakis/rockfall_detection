import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import topo_ops


class Conv2d(layers.Layer):

    def __init__(self, filters, strides=[1, 1], activation=tf.nn.relu, padding='VALID', initializer='glorot_normal',
                 bn=True, momentum=.99):
        super(Conv2d, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn
        self.momentum = momentum

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name='pnet_conv'
        )

        if self.bn: self.bn_layer = layers.BatchNormalization(momentum=self.momentum)

        super(Conv2d, self).build(input_shape)

    def call(self, inputs, training=True):

        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn: points = self.bn_layer(points, training=training)

        if self.activation: points = self.activation(points)

        return points


class PointNetConvLayer(layers.Layer):
    """ The 1D Convolution Layer used by PointNet feature encoder."""

    def __init__(self, filters, bn=True, momentum=.99):
        """ Constructs a Conv1 layer.

        Note:
            Differently from the standard Keras Conv1D layer, the order of ops is:
            1. 1D convolution layer
            2. Batch normalization layer (momentum)
            3. ReLU activation unit

        Input:
            filters: the number of generated features
            momentum: the momentum of the batch normalization layer
        """
        super(PointNetConvLayer, self).__init__()
        self.conv = layers.Conv1D(filters, kernel_size=1, padding="valid")
        self.bn = bn
        self.bn_layer = layers.BatchNormalization(momentum=momentum)

    def call(self, input_tensor, training=False):
        """ Executes the convolution.

        Input:
            inputs: a tensor of size [N, 3].
            training: flag to control batch normalization update statistics.

        Returns:
            Tensor with shape [N, filters].
            """
        x = self.conv(input_tensor)
        if self.bn: x = self.bn_layer(x, training=training)
        x = tf.nn.relu(x)
        return x


class PointNetDenseLayer(layers.Layer):
    """ The fully connected layer used by PointNet classification module.

    Note:
        Differently from the standard Keras Dense layer, the order of ops is:
        1. Fully connected layer
        2. Batch normalization layer (momentum)
        3. ReLU activation unit
        """

    def __init__(self, filters, bn=True, momentum=.99):
        super(PointNetDenseLayer, self).__init__()
        self.dense = layers.Dense(filters)
        self.bn = bn
        self.bn_layer = layers.BatchNormalization(momentum=momentum)

    def call(self, input_tensor, training=False):
        """ Executes the convolution.

        Input:
            inputs: a tensor of size [N, D].
            training: flag to control batch normalization update statistics.

        Returns:
            Tensor with shape [N, filters].
        """
        x = self.dense(input_tensor)
        if self.bn: x = self.bn_layer(x, training=training)
        x = tf.nn.relu(x)
        return x


class TNet(layers.Layer):
    """ TNet description.
        feature encoder - affine transformation matrix prediction.
    """

    def __init__(self, num_features, bn=True, momentum=.99, l2reg=.001):
        super(TNet, self).__init__()
        self.num_features = num_features
        self.conv1 = PointNetConvLayer(16, bn=bn, momentum=momentum)
        self.conv2 = PointNetConvLayer(32, bn=bn, momentum=momentum)
        self.conv3 = PointNetConvLayer(64, bn=bn, momentum=momentum)
        self.pooling = layers.GlobalMaxPooling1D()
        self.dense1 = PointNetDenseLayer(32, bn=bn, momentum=momentum)
        self.dense2 = PointNetDenseLayer(16, bn=bn, momentum=momentum)
        self.bias = keras.initializers.Constant(np.eye(num_features).flatten())
        self.reg = self.OrthogonalRegularizer(num_features, l2reg)
        self.dense3 = layers.Dense(self.num_features * self.num_features,
                                   kernel_initializer="zeros",
                                   bias_initializer=self.bias,
                                   activity_regularizer=self.reg)

    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        # x = self.conv2(x, training=training)
        # x = self.conv3(x, training=training)
        x = self.pooling(x)
        # x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x)

        feat_T = layers.Reshape((self.num_features, self.num_features))(x)
        """ Apply affine transformation to input features."""
        return layers.Dot(axes=(2, 1))([input_tensor, feat_T])


class Pointnet_SA(layers.Layer):

    def __init__(
            self, npoint, radius, nsample, mlp, group_all=False, knn=False, use_xyz=True, activation=tf.nn.relu,
            bn=True, momentum=.99
    ):

        super(Pointnet_SA, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.knn = False
        self.use_xyz = use_xyz
        self.activation = activation
        self.bn = bn
        self.momentum = momentum

        self.mlp_list = []

    def build(self, input_shape):

        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(Conv2d(n_filters, activation=self.activation, bn=self.bn, momentum=self.momentum))

        super(Pointnet_SA, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        if self.group_all:
            nsample = xyz.get_shape()[1]
            new_xyz, new_points, idx, grouped_xyz = topo_ops.sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = topo_ops.sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
                self.knn,
                use_xyz=self.use_xyz
            )

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points = mlp_layer(new_points, training=training)

        new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

        return new_xyz, tf.squeeze(new_points)


class Pointnet_SA_MSG(layers.Layer):

    def __init__(
            self, npoint, radius_list, nsample_list, mlp, use_xyz=True, activation=tf.nn.relu, bn=True, momentum=.99
    ):

        super(Pointnet_SA_MSG, self).__init__()

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp = mlp
        self.use_xyz = use_xyz
        self.activation = activation
        self.bn = bn
        self.momentum = momentum

        self.mlp_list = []

    def build(self, input_shape):

        for i in range(len(self.radius_list)):
            tmp_list = []
            for i, n_filters in enumerate(self.mlp[i]):
                tmp_list.append(Conv2d(n_filters, activation=self.activation, bn=self.bn, momentum=self.momentum))
            self.mlp_list.append(tmp_list)

        super(Pointnet_SA_MSG, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        new_xyz = topo_ops.gather_point(xyz, topo_ops.farthest_point_sample(self.npoint, xyz))

        new_points_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = topo_ops.query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = topo_ops.group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])

            if points is not None:
                grouped_points = topo_ops.group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            for i, mlp_layer in enumerate(self.mlp_list[i]):
                grouped_points = mlp_layer(grouped_points, training=training)

            new_points = tf.math.reduce_max(grouped_points, axis=2)
            new_points_list.append(new_points)

        new_points_concat = tf.concat(new_points_list, axis=-1)

        return new_xyz, new_points_concat


class Pointnet_FP(layers.Layer):

    def __init__(
            self, mlp, activation=tf.nn.relu, bn=True, momentum=.99
    ):

        super(Pointnet_FP, self).__init__()

        self.mlp = mlp
        self.activation = activation
        self.bn = bn
        self.momentum = momentum

        self.mlp_list = []

    def build(self, input_shape):

        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(Conv2d(n_filters, activation=self.activation, bn=self.bn, momentum=self.momentum))
        super(Pointnet_FP, self).build(input_shape)

    def call(self, xyz1, xyz2, points1, points2, training=True):

        if points1 is not None:
            if len(points1.shape) < 3:
                points1 = tf.expand_dims(points1, axis=0)
        if points2 is not None:
            if len(points2.shape) < 3:
                points2 = tf.expand_dims(points2, axis=0)

        dist, idx = topo_ops.three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = topo_ops.three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points1 = mlp_layer(new_points1, training=training)

        new_points1 = tf.squeeze(new_points1)
        if len(new_points1.shape) < 3:
            new_points1 = tf.expand_dims(new_points1, axis=0)

        return new_points1
