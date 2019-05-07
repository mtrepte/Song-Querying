import json
import random

import numpy as np
import tensorflow as tf

from parameters import *
from data import *
from model import *
from log import *
from load import *
from save import *

np.set_printoptions(linewidth=250)

train_x, train_y, test_x, test_y, train_embs, test_embs, train_names, test_names = get_data()

global_step = tf.Variable(0, trainable=False, name='global_step')

model = Model(global_step)

print_log(desc, params=True)
params_file.close()

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=1) 

accs = []; losses = []; precisions = []; recalls = []
for step in range(num_steps):

    rand_idx = np.random.randint(len(train_x), size=batch_size)
    batch_x = train_x[rand_idx]; batch_y = train_y[rand_idx]; batch_embs = train_embs[rand_idx]

    sess.run(tf.local_variables_initializer())
    ops = [model.acc, model.precision, model.recall, model.loss, model.preds, model.train]
    acc, precision, recall, loss, preds, _ = sess.run(ops, feed_dict={model.x: batch_x, model.y: batch_y, model.usage_embs: batch_embs, model.dropout_rate: dropout_rate})

    if step % display_step == 0:
        rand_idx = np.random.randint(len(test_x), size=1024)
        test_batch_x = test_x[rand_idx]; test_batch_y = test_y[rand_idx]; test_batch_embs = test_embs[rand_idx]

        sess.run(tf.local_variables_initializer())
        ops = [model.acc, model.precision, model.recall, model.loss, model.cross_entropy, model.L2_reg, model.usage_loss, model.preds]
        test_acc, test_precision, test_recall, test_loss, cross_entropy_loss, l2_reg_loss, usage_loss, test_preds = sess.run(ops, feed_dict={model.x: test_batch_x, model.y: test_batch_y, model.usage_embs: test_batch_embs})

        losses.append(test_loss); accs.append(test_acc); precisions.append(test_precision); recalls.append(test_recall)
        running_acc = np.mean(accs[-20:]); running_precision = np.mean(precisions[-20:]); running_recall = np.mean(recalls[-20:]); running_loss = np.mean(losses[-20:]);

        print_log('train_label_mean:', str(np.around(np.mean(batch_y, axis=0), 2)))
        print_log('train_pred_mean: ', str(np.around(np.mean(preds, axis=0), 2)))
        print_log()
        print_log('test_label_mean: ', str(np.around(np.mean(test_batch_y, axis=0), 2)))
        print_log('test_pred_mean:  ', str(np.around(np.mean(test_preds, axis=0), 2)))
        print_log()

        print_log('total loss:', test_loss, 'ce:', cross_entropy_loss, 'usage:', usage_loss, 'l2:', l2_reg_loss)

        print_log('\n\nStep: %.0f\n Train Acc: %.3f, Train Precision: %.3f, Train Recall: %.3f, Train Loss: %.3f\n Test Acc: %.3f, Test Precision: %.3f, Test Recall: %.3f, Test Loss: %.3f\n Running Test Acc: %.3f, Running Test Precision: %.3f, Running Test Recall %.3f, Running Test Loss: %.3f\n'
            % (step, acc, precision, recall, loss, test_acc, test_precision, test_recall, test_loss, running_acc, running_precision, running_recall, running_loss))
        flush_file()

    if step % save_step == 0 and step > 0:
        save_model(saver, sess)
        save_embs(sess, model, train_x, test_x, train_y, test_y, train_names, test_names)

save_model(saver, sess)

sess.close()

close_file()