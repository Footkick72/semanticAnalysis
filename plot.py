import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.manifold import TSNE

def load_words(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def project_words(words):
    num_dimensions = 2
    vectors = [words[word] for word in words]
    labels = [word for word in words]

    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    return labels, vectors, vectors_2d

def group_words(words, vectors):
    dim = len(vectors[0])
    centers = [(np.random.rand(dim) * 2 - 1) for _ in range(4)]
    groupings = [[] for _ in range(4)]
    
    for _ in range(10): #iterations

        groupings = [[] for _ in range(4)]
        
        for word in words:
            closest_group = 0
            closest_distance = 1e10
            for i, center in enumerate(centers):
                distance = np.linalg.norm(np.subtract(center, words[word]))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_group = i
            groupings[closest_group].append(word)
        
        for i, group in enumerate(groupings):
            vecs = []
            for word in group:
                vecs.append(np.array(words[word]))
            vecs = np.array(vecs)
            center = np.average(vecs, axis = 0)
            centers[i] = center
    
    return groupings

def plot(labels, vectors_2d, groupings):
    markers = [".", "o", "s", "v"]

    for j, group in enumerate(groupings):
        xs = []
        ys = []
        for word in group:
            xs.append(vectors_2d[labels.index(word)][0])
            ys.append(vectors_2d[labels.index(word)][1])
        plt.scatter(xs, ys, marker = markers[j])

        for i in range(len(group)):
            plt.annotate(
                group[i],
                (vectors_2d[labels.index(group[i])][0], vectors_2d[labels.index(group[i])][1]),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
            )

    print([len(x) for x in groupings])
    plt.show()
    
def main():
    words = load_words("vectors.txt")
    labels, vectors, vectors_2d = project_words(words)
    groupings = group_words(words, vectors)
    plot(labels, vectors_2d, groupings)

if __name__ == "__main__":
    main()