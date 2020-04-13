import numpy as np
import re

def readfile(filename):
    data = []
    with open("data/"+filename) as file:
        for line in file:
            line = line.replace("\n", "") ## remove '\n'
            list_of_str = re.split(' ', line) ## split by space
            list_of_float = []
            for i in range(1, len(list_of_str)):
                list_of_float.append(list_of_str[i]) ## convert to float
            data.append(list_of_float)
    dataset = np.array(data, dtype="float64")
    return dataset