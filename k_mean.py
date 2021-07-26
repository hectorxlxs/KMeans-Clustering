import random
import numpy as np
from math import dist
import matplotlib.pyplot as plt


"""
This class can solve KMeans problem for any number of dimensions,
but only can visualize the data if there are 2 dimensions.
"""


class KMean:

    def __init__(self, centroid_n: int, n_dim: int = 2):
        self.centroid_n = centroid_n
        self.n_dim = n_dim
        self.sse = None
        self.centroids = self.get_rand_centroids()
        self.min_centroid = []
        self.historic = []
        self.dataset = None

    def get_rand_centroids(self):
        return np.random.rand(self.centroid_n, self.n_dim)

    def train(self, dataset, loops: int, max_iter: int):

        """
        :param dataset: an np.array that contains data with shape (data_numbers, n_dims)
        :param loops: represents how many times centroids are randomly chosen to find the best of results
        :param max_iter: represents max of iterations for each loop
        :return: None
        """

        self.dataset = dataset

        self.historic = []

        for loop in range(loops):

            min_centroid = []
            centroids = self.get_rand_centroids()

            for rep in range(max_iter):

                # This variable is saving distance for each point to each centroid
                centroids_dist = []
                for data_index in range(len(dataset)):
                    data_dist = []
                    for i in range(self.centroid_n):
                        data_dist.append(dist(dataset[data_index], centroids[i]))
                    centroids_dist.append(data_dist)

                # Now i group data by closest centroid
                aux_min_centroid = [np.argmin(centroids_dist[i]) for i in range(len(centroids_dist))]
                
                # This means that points are not changing of its associated centroid
                if min_centroid == aux_min_centroid:
                    break

                # Now i want to check if all centroids has at least one point associated
                valid_centroids = True

                for c_index in range(self.centroid_n):
                    if c_index not in aux_min_centroid:
                        valid_centroids = False
                        break

                if not valid_centroids:
                    break

                min_centroid = aux_min_centroid
                centroid_points = []
                centroids_distance = []

                # assigning centroids to the middle of all of its associated points
                for c_index in range(self.centroid_n):

                    # takes all points that correspond to this centroid
                    centroid_points.append(np.array([dataset[i] for i in range(len(min_centroid))
                                                     if min_centroid[i] == c_index]))

                    # calculate middle point of all centroid points
                    point = []
                    for dim_index in range(self.n_dim):
                        point.append(np.mean(centroid_points[c_index][:, dim_index]))
                    centroids[c_index] = np.array(point)

            if len(min_centroid) != 0:

                # Now i'm calculating the distance of each point to its associated centroid.
                centroids_dist = []
                for c_index in range(self.centroid_n):
                    for data_index in range(len(dataset)):

                        # Means that this data does not correspond to this centroid
                        if min_centroid[data_index] != c_index:
                            continue

                        # The distance is kept squared because that gives more importance to the farthest points.
                        centroids_dist.append(dist(centroids[c_index], dataset[data_index])**2)

                sse = np.sum(centroids_dist)

                if self.sse is None or sse < self.sse:
                    self.historic.append(sse)
                    self.sse = sse
                    self.centroids = centroids
                    self.min_centroid = min_centroid

        if len(self.historic) == 0:
            raise ValueError("Not founded valid centroids, try to increase loops parameter")

    def show(self):

        if self.dataset is None:
            raise ValueError("KMean object must be trained before show func")

        if self.n_dim != 2:
            print("I can't visualize more or less than 2 dimensions")

        for c_index in range(self.centroid_n):
            c_data = np.array([self.dataset[i] for i in range(len(self.min_centroid)) if self.min_centroid[i] == c_index])

            color = np.random.rand(3, )
            plt.scatter(c_data[:, 0], c_data[:, 1], color=color)
            plt.scatter(self.centroids[c_index][0], self.centroids[c_index][1], color=color, marker="*")

        plt.show()

    def show_historic(self):
        plt.scatter(range(len(self.historic)), self.historic)
        plt.show()
