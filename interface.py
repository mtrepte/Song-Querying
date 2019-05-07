import numpy as np

def cosine_similarity(vA, vB):
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))


song_embeddings = [np.array([])]
name_to_embedding = {}
embedding_to_name = {}
category_embeddings = {
    "genre1":np.array([])
}

print('seed_song - category_1 + category_2 ~= queried_songs\n')
seed = input("Supply a seed song:")
print("Categories:" + str(category_embeddings.keys()))
cat_1 = input("Supply a category_1")
cat_2 = input("Supply a category_2")

try:
    seed = name_to_embedding[seed]
except:
    print("invalid seed song supplied")

try:
    cat_1 = category_embeddings[cat_1]
    cat_2 = category_embeddings[cat_2]

except:
    print("invalid category supplied")

result = seed - cat_1 + cat_2
result = result / np.linalg.norm(result)

similarity_list = sorted(song_embeddings, key=lambda x: cosine_similarity(x, result))
for embedding in similarity_list[0:5]:
    print(embedding_to_name[embedding])