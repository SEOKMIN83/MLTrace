"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.

"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import os
import sys

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
# plt.figure(figsize=(9 * 2 + 3, 13))
# plt.subplots_adjust(
#     left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
# )

# plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        noisy_circles,  ## data set 1
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,  ## data set 2
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,  ## data set 3
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,  ## data set 4
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),  ## data set 5
    (no_structure, {}),  ## data set 6
]

clustering_dataset_dict = {
    1: noisy_circles,
    2: noisy_moons,
    3: varied,
    4: aniso,
    5: blobs,
    6: no_structure
}

clustering_algo_params_dict = {
    1: {
        "damping": 0.77,
        "preference": -240,
        "quantile": 0.2,
        "n_clusters": 2,
        "min_samples": 7,
        "xi": 0.08,
    },
    2: {
        "damping": 0.75,
        "preference": -220,
        "n_clusters": 2,
        "min_samples": 7,
        "xi": 0.1,
    },
    3: {
        "eps": 0.18,
        "n_neighbors": 2,
        "min_samples": 7,
        "xi": 0.01,
        "min_cluster_size": 0.2,
    },
    4: {
        "eps": 0.15,
        "n_neighbors": 2,
        "min_samples": 7,
        "xi": 0.1,
        "min_cluster_size": 0.2,
    },
    5: {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2},
    6: {}
}

# for i_dataset, (dataset, algo_params) in enumerate(datasets):
#     # update parameters with dataset-specific values
#     params = default_base.copy()
#     params.update(algo_params)
#
#     X, y = dataset
#
#     # normalize dataset for easier parameter selection
#     X = StandardScaler().fit_transform(X)
#
#     # estimate bandwidth for mean shift
#     bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
#
#     # connectivity matrix for structured Ward
#     connectivity = kneighbors_graph(
#         X, n_neighbors=params["n_neighbors"], include_self=False
#     )
#     # make connectivity symmetric
#     connectivity = 0.5 * (connectivity + connectivity.T)
#
#     # ============
#     # Create cluster objects
#     # ============
#     ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
#     ward = cluster.AgglomerativeClustering(
#         n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
#     )
#     spectral = cluster.SpectralClustering(
#         n_clusters=params["n_clusters"],
#         eigen_solver="arpack",
#         affinity="nearest_neighbors",
#     )
#     dbscan = cluster.DBSCAN(eps=params["eps"])
#     optics = cluster.OPTICS(
#         min_samples=params["min_samples"],
#         xi=params["xi"],
#         min_cluster_size=params["min_cluster_size"],
#     )
#     affinity_propagation = cluster.AffinityPropagation(
#         damping=params["damping"], preference=params["preference"], random_state=0
#     )
#     average_linkage = cluster.AgglomerativeClustering(
#         linkage="average",
#         affinity="cityblock",
#         n_clusters=params["n_clusters"],
#         connectivity=connectivity,
#     )
#     birch = cluster.Birch(n_clusters=params["n_clusters"])
#     gmm = mixture.GaussianMixture(
#         n_components=params["n_clusters"], covariance_type="full"
#     )
#
#     clustering_algorithms = (
#         ("MiniBatch\nKMeans", two_means),
#         ("Affinity\nPropagation", affinity_propagation),
#         ("MeanShift", ms),
#         ("Spectral\nClustering", spectral),
#         ("Ward", ward),
#         ("Agglomerative\nClustering", average_linkage),
#         ("DBSCAN", dbscan),
#         ("OPTICS", optics),
#         ("BIRCH", birch),
#         ("Gaussian\nMixture", gmm),
#     )
#
#     for name, algorithm in clustering_algorithms:
#         # t0 = time.time()
#
#         # catch warnings related to kneighbors_graph
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 "ignore",
#                 message="the number of connected components of the "
#                         + "connectivity matrix is [0-9]{1,2}"
#                         + " > 1. Completing it to avoid stopping the tree early.",
#                 category=UserWarning,
#             )
#             warnings.filterwarnings(
#                 "ignore",
#                 message="Graph is not fully connected, spectral embedding"
#                         + " may not work as expected.",
#                 category=UserWarning,
#             )
#             os.system('callgrind_control -i on')
#             algorithm.fit(X)  ## training
#             os.system('callgrind_control -i off')
#
#         # t1 = time.time()
#         ### test : prediction
#         # if hasattr(algorithm, "labels_"):
#         #     y_pred = algorithm.labels_.astype(int)
#         # else:
#         #     y_pred = algorithm.predict(X)
#
#         plot_num += 1

if __name__ == '__main__':

    import os
    import sys

    '''
    sys.argv[1] - clustering algorithm : 1 - two_means. 2 - affinity_propagation, 
    sys.argv[2] - data set : 1 -  (
        noisy_circles,  ## data set 1
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    '''
    params = default_base.copy()
    algo_params = clustering_algo_params_dict[int(sys.argv[2])]
    params.update(algo_params)

    X, y = clustering_dataset_dict[int(sys.argv[2])]

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"], preference=params["preference"], random_state=0
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )

    clustering_algo_dict = {1: two_means,
                            2: affinity_propagation,
                            3: ms,
                            4: spectral,
                            5: ward,
                            6: average_linkage,
                            7: dbscan,
                            8: optics,
                            9: birch,
                            10: gmm}

    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
            category=UserWarning,
        )
        print('algorithm: %s, data set: %s, pid is %s' % (sys.argv[1], sys.argv[2], os.getpid()))

        os.system('callgrind_control -i on')
        clustering_algo_dict[int(sys.argv[1])].fit(X)  ## training
        os.system('callgrind_control -i off')

        # print("END")
        # if hasattr(clustering_algo_dict[sys.argv[1]], "labels_"):
        #     y_pred = clustering_algo_dict[sys.argv[1]].labels_.astype(int)
        # else:
        #     y_pred = clustering_algo_dict[sys.argv[1]].predict(X)
        #
