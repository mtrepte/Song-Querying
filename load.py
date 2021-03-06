import json
import random

import numpy as np
import librosa

from parameters import *
from data import *
from log import *

def get_data_fast():
    data_path = '../datasets/final/'
    with open(data_path + 'finalspecsnames.txt') as f:
        names = f.read().splitlines()
        print(len(names), 'names loaded')
    labels = np.load(data_path + 'finallabels.npy')
    print(labels.shape, 'labels loaded')
    embs = np.load(data_path + 'finalembs.npy')
    print(embs.shape[0], 'embs loaded')
    songs = np.memmap(data_path + 'specs.dat', dtype='float32', mode='r', shape=(len(names), 256, 200))
    songs = songs[:,:,:,np.newaxis]

    if log_spectrograms:
        #songs = np.log(songs + 1e-7)
        songs = librosa.power_to_db(songs)
    if standard_normalize:
        songs = (songs - np.mean(songs, axis=0)) / (np.std(songs, axis=0) + 1e-7)
    #print(np.mean(songs, axis=0))
    #print(np.std(songs, axis=0))    
    return songs, labels, names, embs

def get_data(fast=True):
    if not fast:
        songs, labels, names = get_all_data()
        songs, labels, names = filter_data(songs, labels, names)
        songs, labels, names = preprocess_data(songs, labels, names)
        embs = get_usage_embs(names)
    else:
        songs, labels, names, embs = get_data_fast()

    index = int(len(songs) * train_percentile)
    #index = 100
    train_x = songs[:index]
    train_y = labels[:index]
    train_embs = embs[:index]
    train_names = names[:index]
    test_x = songs[index:]
    test_y = labels[index:]
    test_embs = embs[index:]
    test_names = names[index:]

    return train_x, train_y, test_x, test_y, train_embs, test_embs, train_names, test_names

def get_all_data():
    # data_path = 'data/datasets/goodspecs/'
    # data_path = 'data/datasets/all/'
    data_path = 'data/datasets/sample/'
    # data_path = 'data/datasets/newspecs/'
    labels_path = 'data/song_to_label.json'

    count = 0
    with open(labels_path) as f:
        song_label_pairs = json.load(f)
    song_label_pairs = list(song_label_pairs.items())

    songs = []; labels = []; names = []
    for song_name, label in song_label_pairs:
        try:
            song = np.load(data_path + song_name + '.npy')
            songs.append(song)
            labels.append(label)
            names.append(song_name)
        except:
            pass
        if count % 10000 == 0:
            print('loaded', count, '/', len(song_label_pairs))
        count += 1

    return songs, labels, names

def filter_data(songs, labels, names):
    cutoff_length = 200
    label_counts = np.zeros(num_classes)

    filtered_songs = []; filtered_labels = []; filtered_names = []
    for i in range(len(songs)):
        song = songs[i]; label = labels[i]; name = names[i]
        label = np.array(eval(label))
        if song.shape[1] >= cutoff_length:
            song = song[:,:cutoff_length]
            filtered_songs.append(song)
            filtered_labels.append(label)
            filtered_names.append(name)
            label_counts += label

    print_log('class distribution:', label_counts)

    songs = np.array(filtered_songs)
    songs = songs[:,:,:,np.newaxis]
    labels = np.array(filtered_labels)
    names = np.array(filtered_names)

    return songs, labels, names

def preprocess_data(songs, labels, names):
    if log_spectrograms:
        songs = np.log(songs + 1e-7)
    if standard_normalize:
        songs = (songs - np.mean(songs, axis=0)) / (np.std(songs, axis=0) + 1e-7)

    perm = np.random.permutation(len(songs))
    songs = songs[perm]
    labels = labels[perm]
    names = names[perm]

    print('using', len(songs), 'songs')

    return songs, labels, names

def get_usage_embs(names):
    emb_path = 'data/song_to_usage_emb.json'

    if not train_with_usage_embs:
        return np.zeros((len(names), 32), dtype=np.float32)

    with open(emb_path) as f:
        song_to_usage_emb = json.load(f)

    embs = []
    for song_name in names:
        emb = eval(song_to_usage_emb[song_name])
        embs.append(emb)
    embs = np.array(embs)

    return embs
