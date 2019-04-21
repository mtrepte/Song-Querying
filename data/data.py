import cPickle as cP
import os
import json

def get_id_to_mp3():
	id_to_mp3 = cP.load(open('data/MSD_id_to_7D_id.pkl', 'r'))
	return id_to_mp3

def get_mp3_to_id():
	id_to_mp3 = get_id_to_mp3()
	mp3_to_id = {v: k for k, v in id_to_mp3.iteritems()}
	return mp3_to_id

def get_id_to_tag():
	id_to_tag = cP.load(open('data/msd_id_to_tag_vector.cP','r')) 
	return id_to_tag

def get_id_to_latent_factor():
	TPS_id_to_last_fm_id = TPS_id_to_last_fm_id()
	# TODO: finish

def get_id_to_song():
	file = open('data/unique_tracks.txt', 'r')
	id_to_song = {}

	for line in file:
		last_fm_id, TPS_id, artist, song = line.split('<SEP>')
		id_to_song[last_fm_id] = song
	file.close()

	return id_to_song

def get_TPS_id_to_last_fm_id():
	file = open('data/unique_tracks.txt', 'r')
	TPS_id_to_last_fm_id = {}

	for line in file:
		last_fm_id, TPS_id, artist, song = line.split('<SEP>')
		TPS_id_to_last_fm_id[TPS_id] = last_fm_id
	file.close()

	return TPS_id_to_last_fm_id

def get_all_ids():
	TPS_id_to_last_fm_id = get_TPS_id_to_last_fm_id()

	num_songs = 100000000
	with open('data/usage_data.txt', 'r') as f:
		TPS_ids = []
		for i, line in enumerate(f):
			if i > num_songs:
				break
			user_id, TPS_id, play_count = line.split("\t")
			last_fm_id = TPS_id_to_last_fm_id[TPS_id]
			TPS_ids.append(last_fm_id)
		
	with open('data/msd_id_to_tag_vector.cP', 'r') as f:
		last_fm_ids = []
		for i, line in enumerate(f):
			if i > num_songs:
				break
			if line[:4] == 'tbsS':
				last_fm_id = line[5:-2]
				last_fm_ids.append(last_fm_id)

	# print('TPS_ids:', TPS_ids[:10])
	# print('last_fm_ids:', last_fm_ids[:10])

	TPS_ids = set(TPS_ids)
	last_fms_ids = set(last_fm_ids)
	print 'num songs with usage data:', str(len(TPS_ids))
	print 'num songs with audio tags:', str(len(last_fm_ids))

	intersected_ids = set(TPS_ids).intersection(set(last_fm_ids))
	print 'num songs with both:', str(len(intersected_ids))

	return intersected_ids

def get_song_paths():
	mp3_to_id = get_mp3_to_id()
	ids = get_all_ids()

	src_path = '/Volumes/My Passport/songs'
	dst_path = '.'

	count = 0
	song_paths = []
	for root, dirs, files in os.walk(src_path):
		path = root.split(os.sep)
		print 'path:', root, 'dirs:', dirs
		for file in files:
			if file[-4:] == '.mp3':
				key = file[:-9]
				key = key.replace('_', '')
				key = key.replace('.', '')
				song_id = mp3_to_id[key]
				if song_id in ids:
					print 'count: ', count 
					count += 1
					song_paths.append(root+'/'+file)

	return song_paths

def create_song_path_file():
	paths = get_song_paths()

	with open('data/song_paths.txt', 'w') as f:
		for path in paths:
			f.write(path+'\n')


def get_genre_ids_to_songs():
	ids = get_all_ids()
	id_to_tag = get_id_to_tag()

	genres = [4, 9, 33, 0, 20, 11]
	genre_count = {}
	genre_id_to_songs = {}
	count = 0
	for id in ids:
		tag = id_to_tag[id]
		has_match = False
		valid = False
		matched_genre = -1
		for genre in genres:
			if tag[genre]:
				matched_genre = genre
				if has_match:
					valid = False
				else:
					valid = True
				has_match = True

		if valid:
			genre_count[matched_genre] = genre_count.get(matched_genre, 0) + 1
			genre_id_to_songs[matched_genre] = genre_id_to_songs.get(matched_genre, []) + [id]
		count += 1

	# Remove half of the rock songs
	genre_count[0] //= 2
	genre_id_to_songs[0] = genre_id_to_songs[0][:genre_count[0]]

	print('genre counts:', genre_count)

	return genre_id_to_songs

def save_genre_songs_to_ids():
	path = 'sample/'

	genre_id_to_songs = get_genre_ids_to_songs()
	ids_to_mp3s = get_id_to_mp3()

	genre_songs_to_id = {}
	for id in genre_id_to_songs:
		genre_songs = genre_id_to_songs[id]
		for song in genre_songs:
			mp3 = ids_to_mp3s[song]
			filename = mp3 + '.mpy'
			genre_songs_to_id[path + filename] = id

	filename = 'genre_song_labels.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(genre_songs_to_id, f)


save_genre_songs_to_ids()
