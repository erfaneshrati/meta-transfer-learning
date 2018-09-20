"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes,name='dense_classifier')
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op_metalearner = optimizer(**optim_kwargs).minimize(self.loss)
        with tf.variable_scope('dense_classifier', reuse=True):
            w = tf.get_variable('kernel')
            b = tf.get_variable('bias')
        self.w_zero_op = w.assign(tf.zeros(w.shape))
        self.b_zero_op = b.assign(tf.zeros(b.shape))
class MiniImageNetMetaTransferModel:
    """
    A model for Meta Transfer Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.embedding = out
        self.logits_metalearner = tf.layers.dense(out, num_classes, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(),name='dense_metalearner')
        self.logits_classifier = tf.layers.dense(out, 64, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(),name='dense_classifier')
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.real_label = tf.placeholder(tf.int32, shape=(None,))
        self.loss_metalearner = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits_metalearner)
        self.loss_classifier = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_label,
                                                                   logits=self.logits_classifier)
        self.predictions = tf.argmax(self.logits_metalearner, axis=-1)
        self.minimize_op_metalearner = optimizer(**optim_kwargs).minimize(self.loss_metalearner)
        self.minimize_op_classifier = optimizer(**optim_kwargs).minimize(self.loss_classifier)

        with tf.variable_scope('dense_classifier', reuse=True):
            w = tf.get_variable('kernel')
            b = tf.get_variable('bias')
        self.w_zero_op = w.assign(tf.zeros(w.shape))
        self.b_zero_op = b.assign(tf.zeros(b.shape))
