import numpy as np
import time
import reader
import kmeans

def main():
    ###################### configuration ######################
    k = 4 ## hypervalue "K"
    runs = 20 ## how many times should we run to calculate the average precison/recall/fscore
    time_wait = 0.2 ## because Python pseudo random number generator is time-dependent
    ###########################################################
    dataset_animals = reader.readfile("animals")
    dataset_countries = reader.readfile("countries")
    dataset_fruits = reader.readfile("fruits")
    dataset_veggies = reader.readfile("veggies")
    labels_animals = np.full([dataset_animals.shape[0], 1], 1) ## animals labeled as 1
    labels_countries = np.full([dataset_countries.shape[0], 1], 2) ## countries labeled as 2
    labels_fruits = np.full([dataset_fruits.shape[0], 1], 3) ## fruits labeled as 3
    labels_veggies = np.full([dataset_veggies.shape[0], 1], 4) ## veggies labeled as 4
    dataset = np.vstack([dataset_animals, dataset_countries, dataset_fruits, dataset_veggies])
    labels = np.vstack([labels_animals, labels_countries, labels_fruits, labels_veggies]) ## put them together into one matrix
    clustering = kmeans.train(dataset, k)

    precisions = 0
    recalls = 0
    f_scores = 0
    for i in range (0, runs):
        (p, r, f) = kmeans.test(clustering, labels)
        precisions += p
        recalls += r
        f_scores += f
        time.sleep(time_wait)
    precisions /= runs
    recalls /= runs
    f_scores /= runs
    print("precision: " + str(precisions) + '\n')
    print("recall: " + str(recalls) + '\n')
    print("f_score: " + str(f_scores) + '\n')

if __name__ == "__main__":
    main()