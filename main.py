import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from k_mean import KMean

csv = pd.read_csv('income.csv')

# I take data of columns 1 and 2 of csv and parse them to numpy array
points = csv.iloc[:, 1:3].to_numpy(dtype='float64')

# This is another visual and simple dataset to try
# points = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [5, 5], [6, 5], [5, 6], [6, 6]], dtype='float64')

# dim 1 normalization
min_dim1 = np.min(points[:, 0])
max_dim1 = np.max(points[:, 0])
points[:, 0] = (points[:, 0] - min_dim1) / (max_dim1 - min_dim1)

# dim 2 normalization
min_dim2 = np.min(points[:, 1])
max_dim2 = np.max(points[:, 1])
points[:, 1] = (points[:, 1] - min_dim2) / (max_dim2 - min_dim2)

plt.scatter(points[:, 0], points[:, 1])
plt.show()

km = KMean(3)

km.train(points, 100, 100)

km.show()

# ELBOW METHOD
"""
I have 22 data, so i'm going to look the sse parameter with 
max of 11 centroids.
"""

max_centroids = 11

sse_list = []
for centroids_n in range(1, max_centroids + 1):

    print("Trying to find sse when i have {} centroids...".format(centroids_n))

    km = KMean(centroids_n)

    loops = centroids_n**4  # When i raise the number of centroids i want to find, i need more loops to find them
    km.train(points, loops, 100)
    sse_list.append(km.sse)

plt.scatter(range(1, max_centroids + 1), sse_list)
plt.show()
