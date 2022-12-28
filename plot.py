import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_words(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def project_words(words):
    vectors = np.array([words[word] for word in words])
    labels = [word for word in words]

    u, _, _ = np.linalg.svd(vectors)
    vectors_2d = u[:, :2]

    return labels, vectors_2d


def group_words(words, vectors_2d, labels, n_groupings=4):
    centers = [(np.random.rand(2) * 2 - 1) for _ in range(n_groupings)]
    groupings = [[] for _ in range(n_groupings)]

    for _ in range(10):  # iterations
        groupings = [[] for _ in range(n_groupings)]

        for word in words:
            closest_group = 0
            closest_distance = 1e10
            for i, center in enumerate(centers):
                distance = np.linalg.norm(
                    np.subtract(center, vectors_2d[labels.index(word)])
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_group = i
            groupings[closest_group].append(word)

        for i, group in enumerate(groupings):
            vecs = []
            for word in group:
                vecs.append(vectors_2d[labels.index(word)])
            vecs = np.array(vecs)
            center = np.average(vecs, axis=0)
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
        plt.scatter(xs, ys, marker=markers[j])

        for i in range(len(group)):
            plt.annotate(
                group[i],
                (
                    vectors_2d[labels.index(group[i])][0],
                    vectors_2d[labels.index(group[i])][1],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    for group in groupings:
        print(group)

    print([len(x) for x in groupings])
    plt.show()


def query_word(word, labels, vectors_2d):
    word_vector = None

    for i, label in enumerate(labels):
        if label == word:
            word_vector = list(vectors_2d[i])

    if word_vector == None:
        quit("word not found")

    dists = []
    for i, vector in enumerate(vectors_2d):
        dists.append((labels[i], np.linalg.norm(np.subtract(word_vector, vector))))

    dists.sort(key=lambda x: x[1])

    print(f"top 10 closely related words to {word} are:")

    print("word: distance")
    for entry in dists[1:11]:
        print(f"{entry[0]}: {entry[1]}")


def main(args):
    if len(args) == 0:
        words = load_words("vectors2.txt")
        labels, vectors_2d = project_words(words)
        grouping = None
        i = 0
        while not grouping and i < 1000000:
            groupings = group_words(words, vectors_2d, labels, n_groupings=3)
            if all([len(x) >= 5 for x in groupings]):
                grouping = groupings
            i += 1
            if i % 25 == 0:
                print(i)
        plot(labels, vectors_2d, grouping)
    elif args[0] == "query":
        if len(args) != 2:
            quit("invalid number of arguments")
        words = load_words("vectors.txt")
        labels, vectors_2d = project_words(words)
        query_word(args[1].strip().lower(), labels, vectors_2d)
    else:
        quit("invalid argument")


if __name__ == "__main__":
    main(sys.argv[1:])
