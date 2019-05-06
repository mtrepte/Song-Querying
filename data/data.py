import cPickle as cP
import os
import json

def get_id_to_mp3():
	id_to_mp3 = cP.load(open('metadata/MSD_id_to_7D_id.pkl', 'r'))
	return id_to_mp3

def get_mp3_to_id():
	id_to_mp3 = get_id_to_mp3()
	mp3_to_id = {v: k for k, v in id_to_mp3.iteritems()}
	return mp3_to_id

def get_id_to_tag():
	id_to_tag = cP.load(open('metadata/msd_id_to_tag_vector.cP','r')) 
	return id_to_tag

def get_id_to_latent_factor():
	TPS_id_to_last_fm_id = TPS_id_to_last_fm_id()
	# TODO: finish

def get_id_to_song():
	file = open('metadata/unique_tracks.txt', 'r')
	id_to_song = {}

	for line in file:
		last_fm_id, TPS_id, artist, song = line.split('<SEP>')
		id_to_song[last_fm_id] = song
	file.close()

	return id_to_song

def get_TPS_id_to_last_fm_id():
	file = open('metadata/unique_tracks.txt', 'r')
	TPS_id_to_last_fm_id = {}

	for line in file:
		last_fm_id, TPS_id, artist, song = line.split('<SEP>')
		TPS_id_to_last_fm_id[TPS_id] = last_fm_id
	file.close()

	return TPS_id_to_last_fm_id

def get_all_ids():
	TPS_id_to_last_fm_id = get_TPS_id_to_last_fm_id()

	with open('metadata/usage_data.txt', 'r') as f:
		TPS_ids = []
		for i, line in enumerate(f):
			user_id, TPS_id, play_count = line.split("\t")
			last_fm_id = TPS_id_to_last_fm_id[TPS_id]
			TPS_ids.append(last_fm_id)
		
	with open('metadata/msd_id_to_tag_vector.cP', 'r') as f:
		last_fm_ids = []
		for i, line in enumerate(f):
			if line[:4] == 'tbsS':
				last_fm_id = line[5:-2]
				last_fm_ids.append(last_fm_id)

	TPS_ids = set(TPS_ids)
	last_fms_ids = set(last_fm_ids)
	print('num songs with usage data:', str(len(TPS_ids)))
	print('num songs with audio tags:', str(len(last_fm_ids)))

	intersected_ids = set(TPS_ids).intersection(set(last_fm_ids))
	print('num songs with both:', str(len(intersected_ids)))

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
		print('path:', root, 'dirs:', dirs)
		for file in files:
			if file[-4:] == '.mp3':
				key = file[:-9]
				key = key.replace('_', '')
				key = key.replace('.', '')
				song_id = mp3_to_id[key]
				if song_id in ids:
					print('count: ', count )
					count += 1
					song_paths.append(root+'/'+file)

	return song_paths

def create_song_path_file():
	paths = get_song_paths()

	with open('data/song_paths.txt', 'w') as f:
		for path in paths:
			f.write(path+'\n')

def get_genres_to_song_ids():
	ids = get_all_ids()
	id_to_tag = get_id_to_tag()

	genres = [4, 9, 33, 0, 20, 11]
	genre_count = {}
	genre_to_song_ids = {}
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
			genre_to_song_ids[matched_genre] = genre_to_song_ids.get(matched_genre, []) + [id]
		count += 1

	# Remove half of the rock songs
	genre_count[0] //= 2
	genre_to_song_ids[0] = genre_to_song_ids[0][:genre_count[0]]

	print('genre counts:', genre_count)

	return genre_to_song_ids

def save_genre_songs_to_ids():
	path = ''

	genre_to_song_ids = get_genres_to_song_ids()
	ids_to_mp3s = get_id_to_mp3()

	songs_to_genre = {}
	for genre in genre_to_song_ids:
		song_ids = genre_to_song_ids[genre]
		for song_id in song_ids:
			song = ids_to_mp3s[song_id]
			song = song + '.mpy'
			songs_to_genre[path + song] = genre

	filename = 'genre_song_labels.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(songs_to_genre, f)


def get_song_to_usage_emb():
	id_to_usage_emb = cP.load(open('metadata/id_to_usage_emb.pkl', 'r'))
	ids_to_mp3s = get_id_to_mp3()

	song_to_usage_emb = {}
	for song_id in id_to_usage_emb:
		emb = id_to_usage_emb[song_id]
		song = ids_to_mp3s[song_id]
		song = song + '.mpy'

		song_to_usage_emb[song] = str(emb)

	filename = 'usage_embs.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(song_to_usage_emb, f)


get_song_to_usage_emb()