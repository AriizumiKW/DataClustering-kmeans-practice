import numpy as np
import random
from copy import deepcopy

def train(dataset, k, converge_limit=5):
    n = dataset.shape[0]
    dimensionality = dataset.shape[1]
    distances = np.zeros([n, k], dtype="float64") ## distances to centroids of each cluster
    centroids = np.zeros([k, dimensionality]) ## centroids of each cluster
    clustering_before = np.full([n, 1], 5) ## hold info about each instance is assigned into which cluster before update, used to check converge
    clustering = np.full([n, 1], 5) ## hold info about each instance is assigned into which cluster
    converge_counter = 0

    ## randomly select K different points as initial centroids
    random_integers = generate_k_different_random_int(k, n)
    for i in range(0, k):
        centroids[i] = dataset[random_integers[i]]

    ## start iterations
    while True:
        for i, instance in enumerate(dataset):
            for j, centroid in enumerate(centroids):
                distances[i, j] = euclidean_distance(instance, centroid)

            ## find which cluster is the nearest, and join it
            min_distance = distances[i, 0]
            min_distance_cluster_num = 0
            for j in range(0, k):
                distance = distances[i, j]
                if distance < min_distance:
                    min_distance = distance
                    min_distance_cluster_num = j
            clustering[i, 0] = min_distance_cluster_num

        ## update centroids
        for i in range(0, k):
            points = []
            for j, cluster in enumerate(clustering):
                if cluster == i:
                    points.append(dataset[j])
            sum = np.zeros([1, dimensionality])
            for point in points:
                sum = sum + point
            mean = sum / len(points)
            centroids[i] = mean
        ##print((np.sum(clustering==0),np.sum(clustering==1),np.sum(clustering==2),np.sum(clustering==3)))
        ## check converge
        if (clustering == clustering_before).all():
            converge_counter += 1
        clustering_before = deepcopy(clustering)
        if converge_counter >= converge_limit:
            return clustering


def test(clustering, labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    n = len(clustering)
    for i in range(0, n):
        for j in range(i+1, n): # find each pair
            if(clustering[i] == clustering[j] and labels[i] == labels[j]):
                true_positive += 1
            elif(clustering[i] != clustering[j] and labels[i] != labels[j]):
                true_negative += 1
            elif(clustering[i] == clustering[j] and labels[i] != labels[j]):
                false_positive += 1
            elif(clustering[i] != clustering[j] and labels[i] == labels[j]):
                false_negative += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * precision * recall / (precision + recall)
    return (precision, recall, f_score)



def euclidean_distance(instance, centroid):
    return np.sqrt(np.sum(np.square(instance - centroid)))


def generate_k_different_random_int(k, n):
    random_integers = []
    for i in range(0, k):
        while True:
            rand_int = random.randint(0, n-1)
            for integer in random_integers:
                if integer == rand_int: ## dont allow to generate two same centroids
                    continue
            random_integers.append(rand_int)
            break
    return random_integers
