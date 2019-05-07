import numpy as np
import json

def cosine_similarity(vA, vB):
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))

dir = 'final/'
#with open(dir + 'final_song_to_emb.json') as f:
#    song_to_emb = json.load(f)
#
#with open(dir + 'final_category_to_emb.json') as f:
#    category_to_emb = json.load(f)

categories = ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', 'alternative rock', 'jazz', 'metal', 'chillout', 'classic rock', 'soul', 'indie rock', 'electronica', '80s', 'folk', 'instrumental', 'punk', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental', 'Hip-Hop', 'country', 'funk', 'heavy metal', 'Progressive rock', 'rnb']

print("CATEGORIES")
for i, category in enumerate(categories):
    print(str(i) + ": " + category)
print()
print('seed_song - category_1 + category_2 ~= queried_songs\n')


while True:
    seed = input("Supply a seed song:  ")
    try:
        seed = song_to_emb[seed]
    except:
        print("invalid seed song supplied")
        continue

    cat_1 = input("Supply category_1:  ")
    try:
        cat_1 = category_to_emb[cat_1]
    except:
        print("invalid category supplied")
        continue

    cat_2 = input("Supply category_2:  ")
    try:
        cat_2 = category_to_emb[cat_2]
    except:
        print("invalid category supplied")
        continue

    result = seed - cat_1 + cat_2
    result = result / np.linalg.norm(result)
    
    song_emb_pairs = sorted(song_to_emb.items(), key=lambda x: cosine_similarity(x[1], result))
    for song, emb in song_emb_pairs[:8]:
        print(song, 'with cosine distance', cosine_similarity(emb, result))

    
