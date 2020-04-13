import numpy as np
import time
import reader
import kmeans

def main():
    ###################### configuration ######################
    directories = ["data/animals", "data/countries", "data/fruits", "data/veggies"] ## must be ordered to 1.animals 2.countries 3.fruits 4.veggies
    runs = 20 ## how many times should we run to calculate the average precison/recall/f-score
    time_wait = 0.1 ## wait a while before next running, because Python pseudo random number generator is time-dependent
    if_l2_norm = False ## if use L2-normalised feature vectors
    which_way_to_calculate_distance = 1 ## 1: euclidean distance, 2: Manhattan distance, 3: cosine similarity
    ###########################################################
    dataset_animals = reader.readfile(directories[0])
    dataset_countries = reader.readfile(directories[1])
    dataset_fruits = reader.readfile(directories[2])
    dataset_veggies = reader.readfile(directories[3])
    labels_animals = np.full([dataset_animals.shape[0], 1], 1) ## animals labeled as 1
    labels_countries = np.full([dataset_countries.shape[0], 1], 2) ## countries labeled as 2
    labels_fruits = np.full([dataset_fruits.shape[0], 1], 3) ## fruits labeled as 3
    labels_veggies = np.full([dataset_veggies.shape[0], 1], 4) ## veggies labeled as 4
    dataset = np.vstack([dataset_animals, dataset_countries, dataset_fruits, dataset_veggies])
    labels = np.vstack([labels_animals, labels_countries, labels_fruits, labels_veggies]) ## put them together into one matrix

    ks = []
    precisions = []
    recalls = []
    f_scores = []
    for k in range(1, 11): ## 1 to 11, not including 11
        ks.append(k)
        (p, r, f) = run_and_test(k, dataset, labels, runs, time_wait, which_way_to_calculate_distance, if_l2_norm)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)
    kmeans.draw_plot(ks, precisions, recalls, f_scores)


def run_and_test(k, dataset, labels, runs, time_wait, which_way_to_calculate_distance, if_l2_norm):
    precision = 0
    recall = 0
    f_score = 0
    for i in range(0, runs):
        clustering = kmeans.train(dataset, k, which_way_to_calculate_distance, if_l2_norm)
        (p, r, f) = kmeans.test(clustering, labels)
        precision += p
        recall += r
        f_score += f
        time.sleep(time_wait)
    precision /= runs
    recall /= runs
    f_score /= runs
    return (precision, recall, f_score)


if __name__ == "__main__":
    main()