import json
import random

import numpy as np
import librosa

from parameters import *
from data import *
from log import *

def get_data():
    songs, labels = get_all_data()
    songs, labels = filter_data(songs, labels)
    songs, labels = preprocess_data(songs, labels)

    index = int(len(songs) * train_percentile)
    train_x = songs[:index]
    train_y = labels[:index]
    test_x = songs[index:]
    test_y = labels[index:]

    return train_x, train_y, test_x, test_y

def get_all_data():
    data_path = 'data/spectrograms/all/'
    # data_path = 'data/spectrograms/sample/'
    labels_path = 'data/genre_labels.json'
    songs_path = 'data/genre_song_names.txt'

    old_to_new_label = {4:0, 9:1, 33:2, 0:3, 20:4, 11:5}
    count = 0

    valid_songs = set()
    with open(songs_path) as f:
        for line in f:
            valid_songs.add(line.replace('\n', ''))

    with open(labels_path) as f:
        song_label_pairs = json.load(f)
    song_label_pairs = list(song_label_pairs.items())

    songs = []; labels = []
    for song_name, label in song_label_pairs:
        song_name = song_name[:-3] + 'npy'
        if song_name not in valid_songs:
            continue
        try:
            song = np.load(data_path + song_name)
            songs.append(song)
            label = old_to_new_label[label]
            labels.append(label)
        except:
            pass
        if count % 10000 == 0:
            print('loaded', count, '/', len(song_label_pairs))
        count += 1

    print('using', len(songs), 'songs')

    return songs, labels

def filter_data(songs, labels):
    num_in_one_class_cap = 4400
    # num_in_one_class_cap = 50

    cutoff_length = 300
    label_counts = {}

    filtered_songs = []; filtered_labels = []
    for i in range(len(songs)):
        song = songs[i]; label = labels[i]
        if label_counts.get(label, 0) >= num_in_one_class_cap:
            continue
        if label == 5: # Throw-out the least common genre
            continue
        if song.shape[1] >= cutoff_length:
            song = song[:,:cutoff_length]
            filtered_songs.append(song)
            filtered_labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1

    print('class distribution:', label_counts)

    songs = np.array(filtered_songs)
    songs = songs[:,:,:,np.newaxis]
    labels = np.array(filtered_labels)

    return songs, labels

def preprocess_data(songs, labels):
    if log_spectrograms:
        songs = np.log(songs + 1e-7)
    if standard_normalize:
        songs = (songs - np.mean(songs, axis=0)) / (np.std(songs, axis=0) + 1e-7)

    perm = np.random.permutation(len(songs))
    songs = songs[perm]
    labels = labels[perm]

    return songs, labels

def save_model(saver, sess):
    save_dir = log_dir + '/saved/'
    saver.save(sess, save_dir+'model.ckpt')