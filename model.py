import tensorflow as tf
import numpy as np

from parameters import * 

class Model():
    def __init__(self, global_step):
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=(None, 128, 300, 1))
        self.y = tf.placeholder(tf.int64, shape=(None,))

        # First Layer
        w1 = tf.Variable(tf.truncated_normal([4, 128, 1, 256], stddev=0.1, dtype=tf.float32))
        b1 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))

        conv = tf.nn.conv2d(self.x, w1, strides=[1, 10000, 3, 1], padding='SAME')
        print('conv layer shape:', conv.shape)
        relu = tf.nn.relu(tf.nn.bias_add(conv, b1))
        pool = tf.nn.max_pool(relu, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        print('pool layer shape:', pool.shape)

        # Second Layer
        w2 = tf.Variable(tf.truncated_normal([4, 1, 256, 512], stddev=0.1, dtype=tf.float32))
        b2 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32))

        conv = tf.nn.conv2d(pool, w2, strides=[1, 4, 4, 1], padding='SAME')
        print('conv layer shape:', conv.shape)
        relu = tf.nn.relu(tf.nn.bias_add(conv, b2))
        pool = tf.nn.max_pool(relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        print('pool layer shape:', pool.shape)

        # Flatten
        pool_shape = pool.get_shape().as_list()
        flatten_dim = pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshape = tf.reshape(pool, (-1, flatten_dim))
        print('flatten layer shape:', flatten_dim)

        # Third Layer
        w3 = tf.Variable(tf.truncated_normal([flatten_dim, 512], stddev=0.1, dtype=tf.float32))
        b3 = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
        dense = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, w3) + b3), rate=dropout_rate)
        print('dense layer shape:', dense.shape)

        # Fourth Layer
        w3 = tf.Variable(tf.truncated_normal([flatten_dim, 128], stddev=0.1, dtype=tf.float32))
        b3 = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32))
        dense = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, w3) + b3), rate=dropout_rate)
        print('dense layer shape:', dense.shape)

        # Fifth Layer
        w4 = tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.1, dtype=tf.float32))
        b4 = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32))
        logits = tf.matmul(dense, w4) + b4
        print('dense layer shape:', logits.shape, '\n')

        # Accuracy
        self.preds = tf.argmax(logits, axis=1)
        correct_preds = tf.equal(self.preds, self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))

        # Loss
        L2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.loss = cross_entropy + L2_reg

        # Optimization
        self.global_step = global_step
        if decay_LR:
            self.lr = tf.train.exponential_decay(LR, self.global_step, decay_LR_step, .1, staircase=True)
        else:
            self.lr = tf.constant(LR, dtype=tf.float32)
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=beta1, beta2=beta2).minimize(self.loss)





