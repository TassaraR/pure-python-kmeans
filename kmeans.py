'''
Author: Renzo Tassara Miller
'''

import numpy as np
from multiprocessing import Pool
from functools import partial
import math
import itertools


class KMeans:

    def __init__(self, k, init_k=None, processors=1, kind=None,
                 random_state=None, max_iter=300):

        # Init Params
        self._data = None
        self._k = k
        self._max_iter = max_iter
        self._processes = processors
        self._kind = kind
        self._init_k = init_k
        self._init_idx = None
        self._random_state = random_state

        # Results
        self._centers = None
        self._clusters = None
        self._convergence = None

    def _set_init_centers(self, data):

        if self._init_k is None:
            if self._random_state:
                np.random.seed(self._random_state)
            self._init_idx = np.random.choice(range(len(data)), size=self._k, replace=False)

    def _kmeans_numpy(self):
        """
        K-Means implementation using only numpy arrays
        """
        dist = lambda x, y: np.linalg.norm(x-y, axis=1)

        if self._init_idx is not None:
            c_init = self._data[self._init_idx, :]
        else:
            c_init = self._init_k
        c_curr = c_init

        for itr in range(self._max_iter):

            d_matrix = np.stack([dist(c_curr[i, :], self._data) for i in range(self._k)], axis=1)
            clusters = d_matrix.argmin(axis=1)

            c_prev = c_curr
            c_curr = np.stack([self._data[clusters == i].mean(axis=0) for i in range(self._k)], axis=0)

            if (c_prev == c_curr).all():
                break

        self._centers = c_curr
        self._clusters = clusters
        self._convergence = itr

    @staticmethod
    def euc_dist(arr1, arr2):
        """
        Calculates euclidean distance between 2 vectors arr1 and arr2
        """
        return sum([(a - b)**2 for a, b in zip(arr1, arr2)])**(1/2)

    @staticmethod
    def k_euc_dist(data, k):
        """
        Performs euc_dist for each row
        """
        return [KMeans.euc_dist(row, k) for row in data]

    @staticmethod
    def chunk_size(arr, n_div):
        """
        Returns the size of each chunk given a number of desired chunks
        """
        return math.ceil(len(arr) / n_div)

    @staticmethod
    def create_chunk(arr, chunk):
        return [arr[i:i + chunk] for i in range(0, len(arr), chunk)]

    @staticmethod
    def min_row(row):
        """
        Calculates the index with the minimum value in a list
        """
        return row.index(min(row))

    @staticmethod
    def calc_clusters(data, centers):
        """
        Assigns clusters from a distance matrix built between
        distances of data points and each cluster center.
        """
        dist = [KMeans.k_euc_dist(data, k) for k in centers]
        return [row.index(min(row)) for row in zip(*dist)]

    @staticmethod
    def calc_cluster_per_row(row, centers):
        """
        Calculates the cluster for 1 row
        """
        drow = [KMeans.euc_dist(row, c) for c in centers]
        return KMeans.min_row(drow)

    def _kmeans_python(self):

        data_list = self._data.tolist()

        if self._processes > 1:
            workers = Pool(self._processes)
            if self._kind == 'python_chunks':
                chunk_size = KMeans.chunk_size(data_list, self._processes)
                data_chunks = KMeans.create_chunk(data_list, chunk_size)

        if self._init_idx is not None:
            c_init = [i for n, i in enumerate(data_list) if n in self._init_idx]
        else:
            # assumes centers are np.arrays
            c_init = self._init_k.tolist()
        c_curr = c_init

        for itr in range(self._max_iter):

            if self._processes > 1:
                if self._kind == 'python_chunks':
                    res = workers.map(partial(KMeans.calc_clusters, centers=c_curr), data_chunks)
                    # Bit faster than using a list comprehension
                    clusters = list(itertools.chain(*res))
                elif self._kind == 'python_k':
                    res = workers.map(partial(KMeans.k_euc_dist, data_list), c_curr)
                    clusters = [KMeans.min_row(row) for row in zip(*res)]
                elif self._kind == 'python_rows':
                    cclust = partial(KMeans.calc_cluster_per_row, centers=c_curr)
                    clusters = workers.map(cclust, data_list)
                else:
                    raise ValueError
            else:
                clusters = []
                for row in data_list:
                    cluster = KMeans.calc_cluster_per_row(row, c_curr)
                    clusters.append(cluster)

            c_prev = c_curr
            c_curr = []
            for nc in range(self._k):
                c_i = [i for c, i in zip(clusters, data_list) if c == nc]
                c_i_mean = [sum(i)/len(i) for i in zip(*c_i)]
                c_curr.append(c_i_mean)

            if c_prev == c_curr:
                break

        if self._processes > 1:
            workers.close()
            workers.join()

        self._centers = c_curr
        self._clusters = clusters
        self._convergence = itr

    def fit(self, data):
        """
        Runs kmeans scripts
        """
        self._data = data

        if self._init_k is None:
            self._set_init_centers(data)

        if self._kind == 'numpy':
            self._kmeans_numpy()
        else:
            self._kmeans_python()

    def get_init_k(self):
        return self._data[self._init_idx, :]

    @property
    def centers(self):
        return self._centers

    @property
    def clusters(self):
        return self._clusters

    @property
    def convergence(self):
        return self._convergence
