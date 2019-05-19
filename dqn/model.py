import random

import numpy as np
import tensorflow as tf


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # we create placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # remember that target_Q is the R(s,a) + ymax Qhats(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="valid",                                                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

            self.output = tf.layers.dense(inputs=self.fc, units=self.action_size,
                                          activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # the loss is the difference between our predicted Q_values and the Q_target
            # sum(Q_target - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class DQNetworkPrio:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetworkPrio'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # we create placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions")

            # Prioritized Experience Replay Weights
            self.importance_weights_ph_ = tf.placeholder(tf.float32, [None], name="weight")

            # remember that target_Q is the R(s,a) + ymax Qhats(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="valid",                                                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

            self.output = tf.layers.dense(inputs=self.fc, units=self.action_size,
                                          activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            # Q is the predicted q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # the loss is different compared to above due to prioritized experience replay
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.importance_weights_ph_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class DDQNPrio:
    def __init__(self, state_size, action_size, learning_rate, name='DoubleDeepQNetworkPrioReplay'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # we create placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions")

            # Prioritized Experience Replay Weights
            self.importance_weights_ph_ = tf.placeholder(tf.float32, [None], name="weight")

            # remember that target_Q is the R(s,a) + ymax Qhats(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="valid",                                                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                          padding="valid",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            # Here we separate into two streams (DDQN)
            # The one that calculate value: V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc, units=1, activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate action: A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc, units=self.action_size, activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Aggregating layer: Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + \
                tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is the predicted q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # the loss is different compared to above due to prioritized experience replay
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.importance_weights_ph_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
