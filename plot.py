import matplotlib.pyplot as plt
import numpy as np
import json

def load_words(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def project_words(words):
    vectors = np.array([words[word] for word in words])
    labels = [word for word in words]

    u, s, v = np.linalg.svd(vectors)
    vectors_2d = np.transpose(np.matmul(v[0:2,:], np.transpose(vectors)))

    return labels, vectors_2d

def group_words(words, vectors_2d, labels, n_groupings = 4):
    centers = [(np.random.rand(2) * 2 - 1) for _ in range(n_groupings)]
    groupings = [[] for _ in range(n_groupings)]
    
    for _ in range(10): #iterations
        
        groupings = [[] for _ in range(n_groupings)]
        
        for word in words:
            closest_group = 0
            closest_distance = 1e10
            for i, center in enumerate(centers):
                distance = np.linalg.norm(np.subtract(center, vectors_2d[labels.index(word)]))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_group = i
            groupings[closest_group].append(word)
        
        for i, group in enumerate(groupings):
            vecs = []
            for word in group:
                vecs.append(vectors_2d[labels.index(word)])
            vecs = np.array(vecs)
            center = np.average(vecs, axis = 0)
            centers[i] = center
    
    return groupings

def plot(labels, vectors_2d, groupings):
    markers = "ovs*PXDdp."

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
    labels, vectors_2d = project_words(words)
    groupings = group_words(words, vectors_2d, labels, n_groupings=10)
    plot(labels, vectors_2d, groupings)

if __name__ == "__main__":
    main()