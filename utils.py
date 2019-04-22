import json
import random

import numpy as np
import librosa

from parameters import *
from data import *
from log import *

def get_data():
    data_path = 'data/spectrograms/all/'
    labels_path = 'data/labels.json'

    old_to_new_label_map = {4:0, 9:1, 33:2, 0:3, 20:4, 11:5}
    cutoff_length = 300

    with open(labels_path) as f:
        labels_dict = json.load(f)
    songs = []; labels = []; song_names = []

    song_label_pairs = list(labels_dict.items())
    random.shuffle(song_label_pairs)

    count = 0
    for song_name, label in song_label_pairs:
        song_name = song_name[:-3] + 'npy'
        try:
            song = np.load(data_path + song_name)
            if song.shape[1] >= cutoff_length:
                song = song[:,:cutoff_length]
                # song = np.log(song + 1e-7)
                song = (song - np.mean(song)) / (np.std(song)+1e-7)
                songs.append(song)
                label = old_to_new_label_map[label]
                labels.append(label)
                song_names.append(song_name)
        except:
            pass
        if count % 10000 == 0:
            print('loaded', count, '/', len(labels_dict))
        count += 1

    # More filtering to ensure even class distrbution
    filtered_songs = []; filtered_labels = []; filtered_song_names = []
    label_counts = {}
    num_in_one_class_cap = 4400
    # num_in_one_class_cap = 50
    for i in range(len(songs)):
        song = songs[i]
        label = labels[i]
        song_name = song_names[i]
        if label_counts.get(label, 0) >= num_in_one_class_cap:
            continue
        if label == 5: # Throw-out the least common genre
            continue
        label_counts[label] = label_counts.get(label, 0) + 1

        filtered_labels.append(label)
        filtered_songs.append(song)
        filtered_song_names.append(song_name)

    print('data class distribution:', label_counts)

    songs = np.array(filtered_songs)
    songs = songs[:,:,:,np.newaxis] # Add a channel axis
    # mean = np.mean(songs)
    # std = np.std(songs)
    # songs = (songs - mean) / std

    labels = np.array(filtered_labels)
    song_names = np.array(filtered_song_names)

    num_songs = len(songs)
    print('using', str(num_songs), 'songs\n')

    perm = np.random.permutation(len(songs))
    songs = songs[perm]
    labels = labels[perm]
    song_names = song_names[perm]

    print()
    for song_name in song_names:
        print(song_name)
    import sys; sys.exit()

    index = int(num_songs * train_percentile)
    train_x = songs[:index]
    train_y = labels[:index]
    test_x = songs[index:]
    test_y = labels[index:]

    return train_x, train_y, test_x, test_y

def save_model(saver, sess):
    save_dir = log_dir + '/saved/'
    saver.save(sess, save_dir+'model.ckpt')