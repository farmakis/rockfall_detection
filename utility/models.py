import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, './')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout, Dense
from utility import layers


class PointNet(keras.Model):
    def __init__(self, num_points, output_units=1, activation="sigmoid", bn=True, momentum=.99, l2reg=.0001, dropout=.3):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.tnet1 = layers.TNet(3, bn=bn, momentum=momentum, l2reg=l2reg)
        self.conv1 = layers.PointNetConvLayer(16, bn=bn, momentum=momentum)
        self.tnet2 = layers.TNet(16, bn=bn, momentum=momentum, l2reg=l2reg)
        self.conv2 = layers.PointNetConvLayer(16, bn=bn, momentum=momentum)
        # self.conv3 = layers.PointNetConvLayer(32, bn=bn, momentum=momentum)
        # self.conv4 = layers.PointNetConvLayer(64, bn=bn, momentum=momentum)
        self.max_pool = GlobalMaxPooling1D()
        # self.dense1 = layers.PointNetDenseLayer(32, bn=bn, momentum=momentum)
        # self.dropout1 = Dropout(dropout)
        self.dense2 = layers.PointNetDenseLayer(16, bn=bn, momentum=momentum)
        self.dropout2 = Dropout(dropout)
        self.classifier = Dense(output_units, activation=activation)

    def call(self, input_tensor, training=False):
        x = self.tnet1(input_tensor, training=training)
        x = self.conv1(x, training=training)
        x = self.tnet2(x, training=training)
        x = self.conv2(x, training=training)
        # x = self.conv3(x, training=training)
        # x = self.conv4(x, training=training)
        x = self.max_pool(x)
        # x = self.dense1(x, training=training)
        # x = self.dropout1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

    def info(self):
        x = keras.Input(shape=(self.num_points, 3))
        m = keras.Model(inputs=[x], outputs=self.call(x), name="PointNet")
        return m.summary()


class PointNet2_SSG(keras.Model):

    def __init__(self, batch_size, bn=False, momentum=.0, dropout=0.3, activation=tf.nn.relu):
        super(PointNet2_SSG, self).__init__()

        self.activation = activation
        self.batch_size = batch_size
        self.bn = bn
        self.momentum = momentum
        self.keep_prob = dropout

        self.init_network()

    def init_network(self):
        self.layer1 = layers.Pointnet_SA(
            npoint=256, radius=0.1,
            nsample=32,
            mlp=[16, 16],
            group_all=False,
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.layer2 = layers.Pointnet_SA(
            npoint=128,
            radius=0.2,
            nsample=64,
            mlp=[16, 16],
            group_all=False,
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.layer3 = layers.Pointnet_SA(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[16, 16],
            group_all=True,
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.dense1 = Dense(16, activation=self.activation)
        self.dropout1 = Dropout(self.keep_prob)

        # self.dense2 = Dense(16, activation=self.activation)
        # self.dropout2 = Dropout(self.keep_prob)

        self.dense3 = Dense(1, activation=tf.nn.sigmoid)

    def forward_pass(self, input, training):
        xyz, points = self.layer1(input, None, training=training)
        xyz, points = self.layer2(xyz, points, training=training)
        xyz, points = self.layer3(xyz, points, training=training)

        net = tf.reshape(points, (self.batch_size, -1))

        net = self.dense1(net)
        net = self.dropout1(net)

        # net = self.dense2(net)
        # net = self.dropout2(net)

        pred = self.dense3(net)

        return pred

    def train_step(self, input):
        with tf.GradientTape() as tape:
            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):
        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, input, training=False):
        return self.forward_pass(input, training)


class PointNet2_MSG(keras.Model):

    def __init__(self, batch_size, bn=False, momentum=.0, dropout=0.3, activation=tf.nn.relu):
        super(PointNet2_MSG, self).__init__()

        self.activation = activation
        self.batch_size = batch_size
        self.bn = bn
        self.momentum = momentum
        self.keep_prob = dropout

        self.init_network()

    def init_network(self):
        self.layer1 = layers.Pointnet_SA_MSG(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 16],
            mlp=[[16, 16], [16, 16], [16, 16]],
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.layer2 = layers.Pointnet_SA_MSG(
            npoint=128,
            radius_list=[0.3, 0.5],
            nsample_list=[32, 32],
            mlp=[[16, 16], [16, 16], [16, 16]],
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.layer3 = layers.Pointnet_SA(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[16, 16],
            group_all=True,
            activation=self.activation,
            bn=self.bn,
            momentum=self.momentum
        )

        self.dense1 = Dense(16, activation=self.activation)
        self.dropout1 = Dropout(self.keep_prob)

        # self.dense2 = Dense(16, activation=self.activation)
        # self.dropout2 = Dropout(self.keep_prob)

        self.dense3 = Dense(1, activation=tf.nn.sigmoid)

    def forward_pass(self, input, training):
        xyz, points = self.layer1(input, None, training=training)
        xyz, points = self.layer2(xyz, points, training=training)
        xyz, points = self.layer3(xyz, points, training=training)

        net = tf.reshape(points, (self.batch_size, -1))

        net = self.dense1(net)
        net = self.dropout1(net)

        # net = self.dense2(net)
        # net = self.dropout2(net)

        pred = self.dense3(net)

        return pred

    def train_step(self, input):
        with tf.GradientTape() as tape:
            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):
        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, input, training=False):
        return self.forward_pass(input, training)
