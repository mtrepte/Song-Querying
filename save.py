import json
import numpy as np

from parameters import *
from data import *
from log import *


def save_model(saver, sess):
    save_dir = log_dir + '/saved/'
    saver.save(sess, save_dir+'model.ckpt')

def save_category_embs(sess, model, train_x, test_x, train_y, test_y):
    all_songs = np.concatenate([train_x, test_x], axis=0)
    all_labels = np.concatenate([train_y, test_y], axis=0)

    all_embs = sess.run(model.embs, feed_dict={model.x: all_songs, model.y: all_labels})

    # Get 'averaging' category embs
    average_category_embs = []
    for label in range(num_classes):
        embs_with_label = []
        for i, labels in enumerate(all_labels):
            if labels[label]:
                emb = all_embs[i]
                embs_with_label.append(emb)
        category_emb = np.mean(embs_with_label, axis=0)
        average_category_embs.append(category_emb)
    average_category_embs /= np.linalg.norm(average_category_embs, axis=1)[:, np.newaxis]
    # print('should be normalized:', np.linalg.norm(average_category_embs, axis=1))

    # Get 'logits-matrix' category embs
    logits_matrix = sess.run(model.logits_matrix)
    logits_matrix_category_embs = logits_matrix.T
    logits_matrix_category_embs /= np.linalg.norm(logits_matrix_category_embs, axis=1)[:, np.newaxis]
    # print('should be normalized:', np.linalg.norm(logits_matrix_category_embs, axis=1))

    filename = 'category_embs.json'
    category_embs = {'average_category_embs': str(average_category_embs), 'logits_matrix_category_embs': str(logits_matrix_category_embs)}
    save_dir = log_dir + '/saved/'
    with open(save_dir + filename, 'w') as f:
        json.dump(category_embs, f)

