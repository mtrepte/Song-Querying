import tensorflow as tf
import numpy as np

from parameters import * 
from log import *

class Model():
    def __init__(self, global_step):
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=(None, 128, 300, 1))
        self.y = tf.placeholder(tf.int64, shape=(None,))
        self.usage_embs = tf.placeholder(tf.float32, shape=(None, 32))

        # Conv Layer
        w1 = tf.Variable(tf.truncated_normal([8, 8, 1, 64], stddev=0.1, dtype=tf.float32))
        b1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))

        conv = tf.nn.conv2d(self.x, w1, strides=[1, 2, 2, 1], padding='SAME')
        print_log('conv layer shape:', conv.shape)
        pool = tf.nn.max_pool(tf.nn.bias_add(conv, b1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        relu = tf.nn.relu(pool)
        print_log('pool layer shape:', pool.shape)

        # Conv Layer
        w2 = tf.Variable(tf.truncated_normal([5, 5, 64, 96], stddev=0.1, dtype=tf.float32))
        b2 = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32))

        conv = tf.nn.conv2d(relu, w2, strides=[1, 2, 2, 1], padding='SAME')
        print_log('conv layer shape:', conv.shape)
        pool = tf.nn.max_pool(tf.nn.bias_add(conv, b2), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        relu = tf.nn.relu(pool)
        print_log('pool layer shape:', pool.shape)

        # Conv Layer
        w3 = tf.Variable(tf.truncated_normal([3, 3, 96, 128], stddev=0.1, dtype=tf.float32))
        b3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32))

        conv = tf.nn.conv2d(relu, w3, strides=[1, 2, 2, 1], padding='SAME')
        print_log('conv layer shape:', conv.shape)
        pool = tf.nn.avg_pool(tf.nn.bias_add(conv, b3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        relu = tf.nn.relu(pool)
        print_log('pool layer shape:', pool.shape)

        # Flatten
        relu_shape = relu.get_shape().as_list()
        flatten_dim = relu_shape[1]*relu_shape[2]*relu_shape[3]
        reshape = tf.reshape(relu, (-1, flatten_dim))
        print_log('flatten layer shape:', flatten_dim)

        # Dense Layer
        w4 = tf.Variable(tf.truncated_normal([flatten_dim, 256], stddev=0.1, dtype=tf.float32))
        b4 = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32))
        dense = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, w4) + b4), rate=dropout_rate)
        print_log('dense layer shape:', dense.shape)

        # Dense Layer
        w4 = tf.Variable(tf.truncated_normal([flatten_dim, embedding_dim], stddev=0.1, dtype=tf.float32))
        b4 = tf.Variable(tf.constant(0.1, shape=[embedding_dim], dtype=tf.float32))
        dense = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, w4) + b4), rate=dropout_rate)
        print_log('dense layer shape:', dense.shape)
        self.embeddings = tf.nn.l2_normalize(dense, axis=1)

        # Classification Logits Layer
        w5 = tf.Variable(tf.truncated_normal([embedding_dim, num_classes], stddev=0.1, dtype=tf.float32))
        b5 = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32))
        logits = tf.matmul(dense, w5) + b5
        print('logits layer shape:', logits.shape)

        # Usage Emb Layer
        if train_with_usage_embs:
            w6 = tf.Variable(tf.truncated_normal([embedding_dim, 32], stddev=0.1, dtype=tf.float32))
            b6 = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32))
            dense = tf.matmul(dense, w6) + b6
            self.usage_preds = tf.nn.l2_normalize(dense, axis=1)
            print('usage_preds layer shape:', self.usage_preds.shape)

        print()

        # Accuracy
        self.preds = tf.argmax(logits, axis=1)
        correct_preds = tf.equal(self.preds, self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))

        # Loss
        self.L2_reg = L2_reg * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.loss = self.cross_entropy + self.L2_reg
        self.usage_loss = tf.zeros([])
        if train_with_usage_embs:
            normalized_usage_embs = tf.nn.l2_normalize(self.usage_embs, axis=1)
            self.usage_loss = usage_loss_weight * tf.losses.mean_squared_error(normalized_usage_embs, self.usage_preds)
            self.loss +=  self.usage_loss

        # Optimization
        self.global_step = global_step
        if decay_LR:
            self.lr = tf.train.exponential_decay(LR, self.global_step, decay_LR_step, .1, staircase=True)
        else:
            self.lr = tf.constant(LR, dtype=tf.float32)
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=beta1, beta2=beta2).minimize(self.loss)





