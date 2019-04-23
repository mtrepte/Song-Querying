import json
import random

import numpy as np
import tensorflow as tf

from parameters import *
from data import *
from model import *
from log import *
from utils import *


train_x, train_y, test_x, test_y = get_data()

global_step = tf.Variable(0, trainable=False, name='global_step')

model = Model(global_step)

print_log(desc, params=True)
params_file.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=1) 

accs = []; losses = []
for step in range(num_steps):

    rand_idx = np.random.randint(len(train_x), size=batch_size)      
    batch_x = train_x[rand_idx]; 
    batch_y = train_y[rand_idx]

    ops = [model.acc, model.loss, model.preds, model.train]
    acc, loss, preds, _ = sess.run(ops, feed_dict={model.x: batch_x, model.y: batch_y})

    if step % display_step == 0:
        rand_idx = np.random.randint(len(test_x), size=1024)
        test_batch_x = test_x[rand_idx]; 
        test_batch_y = test_y[rand_idx]
        print_log('train_labels:', batch_y[:50])   
        print_log('test_labels:', test_batch_y[:50])

        ops = [model.loss, model.acc, model.preds]
        test_loss, test_acc, test_preds = sess.run(ops, feed_dict={model.x: test_batch_x, model.y: test_batch_y})

        losses.append(test_loss); accs.append(test_acc)
        running_loss = np.mean(losses[-20:]); running_acc = np.mean(accs[-20:])

        print_log('train_preds:', preds[:50])
        print_log('test_preds:', test_preds[:50])
        print_log('\nStep: %.0f\n Train Acc: %.3f, Train Loss: %.3f\n Test Acc: %.3f, Test Loss: %.3f \n Running Test Acc: %.3f, Running Test Loss: %.3f\n' 
            % (step, acc, loss, test_acc, test_loss, running_acc, running_loss))

    if step % save_step == 0:
        save_model(saver, sess)


save_model(saver, sess)

sess.close()

close_file()