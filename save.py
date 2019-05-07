import json
import numpy as np

from log import *


def save_model(saver, sess):
    save_dir = log_dir + '/saved/'
    saver.save(sess, save_dir+'model.ckpt')

def save_embs(sess, model, train_x, test_x, train_y, test_y, train_names, test_names):
    all_songs = np.concatenate([train_x, test_x], axis=0)
    all_labels = np.concatenate([train_y, test_y], axis=0)
    all_names = np.concatenate([train_names, test_names], axis=0)

    all_song_embs = sess.run(model.embs, feed_dict={model.x: all_songs})

    save_dir = log_dir + '/saved/'
    save_song_embs(all_song_embs, all_names, save_dir)
    save_category_embs(sess, model, all_labels, all_song_embs, save_dir)

def save_song_embs(all_song_embs, all_names, save_dir):
    all_song_embs = [str(emb.tolist()) for emb in all_song_embs]
    song_to_emb = dict(zip(all_names, all_song_embs))

    filename = 'song_to_emb.json'
    with open(save_dir + filename, 'w') as f:
        json.dump(song_to_emb, f)

def save_category_embs(sess, model, all_labels, all_song_embs, save_dir):
    # Get 'averaging' category embs
    average_category_embs = []
    for label in range(num_classes):
        embs_with_label = []
        for i, labels in enumerate(all_labels):
            if labels[label]:
                emb = all_song_embs[i]
                embs_with_label.append(emb)
        category_emb = np.mean(embs_with_label, axis=0)
        average_category_embs.append(category_emb)
    average_category_embs /= np.linalg.norm(average_category_embs, axis=1)[:, np.newaxis]
    # print('should be normalized:', np.linalg.norm(average_category_embs, axis=1))

    category_to_average_emb = dict([(i, str(average_category_embs[i].tolist())) for i in range(len(average_category_embs))])


    # Get 'logits-matrix' category embs
    logits_matrix = sess.run(model.logits_matrix)
    logits_matrix_category_embs = logits_matrix.T
    logits_matrix_category_embs /= np.linalg.norm(logits_matrix_category_embs, axis=1)[:, np.newaxis]
    # print('should be normalized:', np.linalg.norm(logits_matrix_category_embs, axis=1))

    category_to_logits_matrix_emb = dict([(i, str(logits_matrix_category_embs[i].tolist())) for i in range(len(logits_matrix_category_embs))])


    category_to_emb = {'category_to_average_emb': category_to_average_emb, 'category_to_logits_matrix_emb': category_to_logits_matrix_emb}

    filename = 'category_to_emb.json'
    with open(save_dir + filename, 'w') as f:
        json.dump(category_to_emb, f)