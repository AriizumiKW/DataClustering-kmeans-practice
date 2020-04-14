import numpy as np
from matplotlib import pyplot as plt


def draw():
    q2 = np.array([
        [1,0.32,1.00,0.49,0.32],
        [2,0.65,1.00,0.79,0.83],
        [3,0.75,0.88,0.81,0.87],
        [4,0.80,0.74,0.77,0.86],
        [5,0.81,0.61,0.70,0.83],
        [6,0.84,0.56,0.67,0.82],
        [7,0.87,0.48,0.62,0.81],
        [8,0.88,0.43,0.58,0.80],
        [9,0.89,0.38,0.53,0.78],
        [10,0.90,0.36,0.52,0.78]])

    q3 = np.array([
        [1,0.32,1.00,0.49,0.32],
        [2,0.65,1.00,0.79,0.83],
        [3,0.78,0.93,0.85,0.89],
        [4,0.82,0.75,0.78,0.87],
        [5,0.84,0.65,0.73,0.85],
        [6,0.88,0.57,0.68,0.83],
        [7,0.89,0.50,0.64,0.82],
        [8,0.90,0.45,0.59,0.80],
        [9,0.91,0.41,0.56,0.79],
        [10,0.93,0.36,0.51,0.78]])

    q4 = np.array([
        [1,0.32,1.00,0.49,0.32],
        [2,0.65,1.00,0.78,0.82],
        [3,0.75,0.88,0.81,0.87],
        [4,0.77,0.73,0.75,0.84],
        [5,0.83,0.62,0.70,0.83],
        [6,0.85,0.55,0.66,0.82],
        [7,0.88,0.47,0.61,0.81],
        [8,0.90,0.44,0.58,0.80],
        [9,0.91,0.40,0.55,0.79],
        [10,0.90,0.37,0.52,0.78]])

    q5 = np.array([
        [1,0.32,1.00,0.49,0.32],
        [2,0.65,1.00,0.79,0.83],
        [3,0.74,0.87,0.80,0.86],
        [4,0.80,0.75,0.77,0.86],
        [5,0.84,0.63,0.71,0.84],
        [6,0.88,0.55,0.67,0.83],
        [7,0.90,0.51,0.64,0.82],
        [8,0.91,0.44,0.59,0.80],
        [9,0.93,0.41,0.56,0.80],
        [10,0.93,0.37,0.52,0.79]])

    q6 = np.array([
        [1,0.32,1.00,0.49,0.32],
        [2,0.65,1.00,0.79,0.83],
        [3,0.66,0.88,0.76,0.81],
        [4,0.67,0.72,0.69,0.79],
        [5,0.66,0.61,0.63,0.77],
        [6,0.67,0.56,0.60,0.77],
        [7,0.67,0.52,0.58,0.76],
        [8,0.65,0.48,0.55,0.75],
        [9,0.64,0.45,0.53,0.74],
        [10,0.64,0.44,0.51,0.74]])

    '''
    ## 7-1
    plt.figure(figsize=(8, 4), dpi=160)
    plt.plot(q2[:, 0], q2[:, 3], color='green', label='f-score (Question2)')
    plt.plot(q4[:, 0], q4[:, 3], color='red', label='f-score (Question4)')
    plt.plot(q6[:, 0], q6[:, 3], color='blue', label='f-score (Question6)')
    plt.plot(q2[:, 0], q2[:, 4], color='yellowgreen', label='rand index (Question2)')
    plt.plot(q4[:, 0], q4[:, 4], color='salmon', label='rand index (Question4)')
    plt.plot(q6[:, 0], q6[:, 4], color='skyblue', label='rand index (Question6)')
    plt.xlabel('k')
    plt.ylabel('P/R/F/RI')
    plt.xticks(range(1, 11, 1))
    plt.legend()
    plt.show()
    '''

    ## 7-2
    plt.figure(figsize=(8, 4), dpi=160)
    ##plt.plot(q2[:, 0], q2[:, 3], color='green', label='f-score (Question2)')
    ##plt.plot(q3[:, 0], q3[:, 3], color='red', label='f-score (Question3)')
    plt.plot(q4[:, 0], q4[:, 3], color='blue', label='f-score (Question4)')
    plt.plot(q5[:, 0], q5[:, 3], color='deeppink', label='f-score (Question5)')
    ##plt.plot(q2[:, 0], q2[:, 4], color='yellowgreen', label='rand index (Question2)')
    ##plt.plot(q3[:, 0], q3[:, 4], color='salmon', label='rand index (Question3)')
    plt.plot(q4[:, 0], q4[:, 4], color='skyblue', label='rand index (Question4)')
    plt.plot(q5[:, 0], q5[:, 4], color='hotpink', label='rand index (Question5)')
    plt.xlabel('k')
    plt.ylabel('P/R/F/RI')
    plt.xticks(range(1, 11, 1))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw()