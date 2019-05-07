import cPickle as cP
import os
import json
from collections import Counter

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

def get_label_counts():
	ids = get_all_ids()
	id_to_tag = get_id_to_tag()

	label_counts = {}
	for song_id in ids:
		tag = id_to_tag[song_id]
		for i, boolean in enumerate(tag):
			if boolean[0]:
				label_counts[i] = label_counts.get(i, 0) + 1

	print(label_counts)

def save_song_to_label():
	ids = get_all_ids()
	id_to_tag = get_id_to_tag()
	id_to_mp3 = get_id_to_mp3()

	# Filter out least common
	#                   0  1  2. 3. 4. 5. 6  7. 8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23  24. 25. 26. 27. 28. 29  30. 31
	valid_labels = set([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16, 18, 19, 20, 23, 24, 26, 27, 28, 29, 30, 33, 36, 40, 42, 43, 45])

	# Clip most common
	threshold = 8000

	label_counts = {}
	song_to_label = {}
	for song_id in ids:
		tag = id_to_tag[song_id]
		one_hot = []
		valid = True
		for label, boolean in enumerate(tag):
			if boolean[0]:
				label_counts[label] = label_counts.get(label, 0) + 1
				if label_counts[label] > threshold:
					valid = False
					break
			if label in valid_labels:
				one_hot.append(int(boolean[0]))

		if valid:
			song = id_to_mp3[song_id]
			song_to_label[song] = str(one_hot)

	final_label_counts = {}
	for one_hot in song_to_label.values():
		for label, boolean in enumerate(eval(one_hot)):
			if boolean:
				final_label_counts[label] = final_label_counts.get(label, 0) + 1

	print('final label dist:', final_label_counts)
	print('num songs:', len(song_to_label))

	filename = 'song_to_label.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(song_to_label, f)



def get_genre_to_ids():
	ids = get_all_ids()
	id_to_tag = get_id_to_tag()
	id_to_mp3 = get_id_to_mp3()


	genres = [4, 9, 33, 0, 20, 11]
	genre_count = {}
	genre_to_ids = {}
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
			genre_to_ids[matched_genre] = genre_to_ids.get(matched_genre, []) + [id]
		count += 1

	# Remove half of the rock songs
	genre_count[0] //= 2
	genre_to_ids[0] = genre_to_ids[0][:genre_count[0]]

	print('genre counts:', genre_count)

	return genre_to_ids

def save_song_to_genre():
	genre_to_id = get_genre_to_ids()
	id_to_mp3 = get_id_to_mp3()

	songs_to_genre = {}
	for genre in genre_to_id:
		song_ids = genre_to_ids[genre]
		for song_id in song_ids:
			song = id_to_mp3[song_id]
			song = song + '.mpy'
			songs_to_genre[path + song] = genre

	filename = 'song_to_genre.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(songs_to_genre, f)


def get_song_to_usage_emb():
	id_to_usage_emb = cP.load(open('metadata/id_to_usage_emb.pkl', 'r'))
	id_to_mp3 = get_id_to_mp3()

	song_to_usage_emb = {}
	for song_id in id_to_usage_emb:
		emb = id_to_usage_emb[song_id]
		song = id_to_mp3[song_id]
		song = song + '.mpy'

		song_to_usage_emb[song] = str(emb)

	filename = 'song_to_usage_embs.json'
	print('saving:', filename)
	with open(filename, 'w') as f:
		json.dump(song_to_usage_emb, f)


save_song_to_label()