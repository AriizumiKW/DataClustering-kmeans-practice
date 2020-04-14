import numpy as np
import random
from copy import deepcopy
from matplotlib import pyplot as plt


def train(data, k, which_way, l2_norm, converge_limit=30):
    dataset = deepcopy(data)
    n = dataset.shape[0]
    dimensionality = dataset.shape[1]
    distances = np.zeros([n, k], dtype="float64")  ## distances to centroids of each cluster
    centroids = np.zeros([k, dimensionality])  ## centroids of each cluster
    clustering_before = np.full([n], 5)  ## hold info about each instance is assigned into which cluster before update, used to check convergence
    clustering = np.full([n], 5)  ## hold info about each instance is assigned into which cluster
    converge_counter = 0
    counter = 0

    if l2_norm:
        dataset = l2_normalise(dataset)
    ## randomly select K different points as initial centroids
    random_integers = generate_k_different_random_int(k, n)
    for i in range(0, k):
        centroids[i] = dataset[random_integers[i]]

    ## start iterations
    while True:
        for i, instance in enumerate(dataset):
            if which_way == 1:
                for j, centroid in enumerate(centroids):
                    distances[i, j] = euclidean_distance(instance, centroid)
            elif which_way == 2:
                for j, centroid in enumerate(centroids):
                    distances[i, j] = manhattan_distance(instance, centroid)
            elif which_way == 3:
                for j, centroid in enumerate(centroids):
                    distances[i, j] = cosine_similarity(instance, centroid)

            ## find which cluster is the nearest, and join it
            min_distance = distances[i, 0]
            min_distance_cluster_num = 0
            for j in range(0, k):
                distance = distances[i, j]
                if distance < min_distance:
                    min_distance = distance
                    min_distance_cluster_num = j
            clustering[i] = min_distance_cluster_num

        ## update centroids
        for i in range(0, k):
            points = dataset[clustering == i]
            if (len(points) != 0):
                mean = np.mean(points, axis=0)
                centroids[i] = mean
        counter += 1
        ## check convergence
        if (clustering == clustering_before).all():
            converge_counter += 1
        clustering_before = deepcopy(clustering)
        if converge_counter >= 4 or counter >= converge_limit: ## force to converge if iterations >= converge_limit
            return clustering


def test(clustering, labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    n = len(clustering)
    for i in range(0, n):
        for j in range(i + 1, n):  # find each pair
            if (clustering[i] == clustering[j] and labels[i] == labels[j]):
                true_positive += 1
            elif (clustering[i] != clustering[j] and labels[i] != labels[j]):
                true_negative += 1
            elif (clustering[i] == clustering[j] and labels[i] != labels[j]):
                false_positive += 1
            elif (clustering[i] != clustering[j] and labels[i] == labels[j]):
                false_negative += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * precision * recall / (precision + recall)
    ri = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return (precision, recall, f_score, ri)


def draw_plot(ks, precisions, recalls, fscores, rand_indices):
    plt.figure(figsize=(8, 4), dpi=160)
    plt.plot(ks, precisions, color='green', label='precision')
    plt.plot(ks, recalls, color='blue', label='recall')
    plt.plot(ks, fscores, color='red', label='f-score')
    plt.plot(ks, rand_indices, color='olive', label='rand index')
    plt.xlabel('k')
    plt.ylabel('P/R/F/RI')
    plt.xticks(range(1, 11, 1))
    plt.legend()
    plt.show()


def l2_normalise(dataset):
    for i, row in enumerate(dataset):
        dataset[i] = row / np.linalg.norm(row)
    return dataset


def euclidean_distance(instance, centroid):
    return np.sqrt(np.sum(np.square(instance - centroid)))


def manhattan_distance(instance, centroid):
    return np.sum(np.abs(instance - centroid))


def cosine_similarity(instance, centroid):
    return np.dot(instance, centroid) / (np.linalg.norm(instance) * np.linalg.norm(centroid))


def generate_k_different_random_int(k, n):
    random_integers = []
    for i in range(0, k):
        while True:
            rand_int = random.randint(0, n - 1)
            for integer in random_integers:
                if integer == rand_int:  ## dont allow to generate two same integers
                    continue
            random_integers.append(rand_int)
            break
    return random_integers
