import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.manifold import TSNE

words = {}
with open("Python/semanticAnalysis/vectors.txt", "r") as f:
    words = json.loads(f.read())

num_dimensions = 2
vectors = [words[word] for word in words]
labels = [word for word in words]

tsne = TSNE(n_components=num_dimensions, random_state=0)
vectors2D = tsne.fit_transform(vectors)

x_vals = [v[0] for v in vectors2D]
y_vals = [v[1] for v in vectors2D]

dim = len(vectors[0])
centers = [(np.random.rand(dim) * 2 - 1) for _ in range(4)]
groupings = [[] for _ in range(4)]
for _ in range(10): #iterations

    groupings = [[] for _ in range(4)]
    
    for word in words:
        closestGroup = 0
        closestDistance = 1e10
        for i, center in enumerate(centers):
            distance = np.linalg.norm(np.subtract(center, words[word]))
            if distance < closestDistance:
                closestDistance = distance
                closestGroup = i
        groupings[closestGroup].append(word)
    
    for i, group in enumerate(groupings):
        vecs = []
        for word in group:
            vecs.append(np.array(words[word]))
        vecs = np.array(vecs)
        center = np.average(vecs, axis = 0)
        centers[i] = center
    print([len(x) for x in groupings])
    
markers = [".", "o", "s", "v"]

for j, group in enumerate(groupings):
    xs = []
    ys = []
    for word in group:
        xs.append(vectors2D[labels.index(word)][0])
        ys.append(vectors2D[labels.index(word)][1])
    plt.scatter(xs, ys, marker = markers[j])

    for i in range(len(group)):
        plt.annotate(
            group[i],
            (vectors2D[labels.index(group[i])][0], vectors2D[labels.index(group[i])][1]),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
        )

print([len(x) for x in groupings])
plt.show()