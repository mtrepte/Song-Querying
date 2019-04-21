import json
from collections import Counter

labels_path = 'data/labels.json'

with open(labels_path) as f:
    labels_dict = json.load(f)

labels = labels_dict.values()

c = Counter(labels)

print(c.keys())
print(c.values())