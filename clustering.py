from collections import Counter
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder

from utils import dist

METRIC = 'wminkowski'
METRIC_PARAMS = {'w': [88.8, 111.0], 'p': 2}

min_dist_between_clusters = 1.2

def plot_xy_with_label(dests, labels, centroid, remove_noise=False, save_dir='./', fname=None):
    color_tuples_ = [
         ('aqua', '#00FFFF'),
         ('blue', '#0000FF'),
         ('blueviolet', '#8A2BE2'),
         ('brown', '#A52A2A'),
         ('burlywood', '#DEB887'),
         ('chartreuse', '#7FFF00'),
         ('chocolate', '#D2691E'),
         ('coral', '#FF7F50'),
         ('cornflowerblue', '#6495ED'),
         ('crimson', '#DC143C'),
         ('deeppink', '#FF1493'),
         ('dodgerblue', '#1E90FF'),
         ('firebrick', '#B22222'),
         ('fuchsia', '#FF00FF'),
         ('gold', '#FFD700'),
         ('goldenrod', '#DAA520'),
         ('green', '#008000'),
         ('greenyellow', '#ADFF2F'),
         ('hotpink', '#FF69B4'),
         ('indianred', '#CD5C5C'),
         ('indigo', '#4B0082'),
         ('lime', '#00FF00'),
         ('limegreen', '#32CD32'),
         ('magenta', '#FF00FF'),
         ('maroon', '#800000'),
         ('navy', '#000080'),
         ('olive', '#808000'),
         ('orange', '#FFA500'),
         ('orangered', '#FF4500'),
         ('palegreen', '#98FB98'),
         ('palevioletred', '#DB7093'),
         ('peru', '#CD853F'),
         ('plum', '#DDA0DD'),
         ('purple', '#800080'),
         ('red', '#FF0000'),
         ('rosybrown', '#BC8F8F'),
         ('royalblue', '#4169E1'),
         ('saddlebrown', '#8B4513'),
         ('salmon', '#FA8072'),
         ('seagreen', '#2E8B57'),
         ('slateblue', '#6A5ACD'),
         ('springgreen', '#00FF7F'),
         ('steelblue', '#4682B4'),
         ('teal', '#008080'),
         ('tomato', '#FF6347'),
         ('turquoise', '#40E0D0'),
         ('violet', '#EE82EE'),
         ('yellowgreen', '#9ACD32')
    ]
    colors_ = [c[0] for c in color_tuples_]

    fig, ax = plt.subplots(figsize=(20,10))
    for label, color in zip(np.unique(labels), colors_):
        if remove_noise and label == -1:
            continue
        class_members = (labels == label)
        legend_name = str(label) + '_' + color
        cluster_size_dict = Counter(labels)
        if cluster_size_dict[label] < 5:
            color = 'gray'
        ax.scatter(dests[class_members, 1], dests[class_members, 0], c=color, alpha=0.2, s=300, marker='.', label=legend_name)
        # if label != -1:
            # centroid = np.mean(dests[class_members], axis=0)
    ax.scatter(centroid[:, 1], centroid[:, 0], marker='+', c='white', s=100, alpha=1)
    plt.title('clustering result')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if fname is None:
        plt.show()
    else:
        save_path = save_dir + '/' + fname
        print(save_path)
        plt.savefig(save_path)
    plt.close()


class ModifiedMeanShift(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.
    cluster frequent points in very small region (e.g., home, office, ...)
    exclude clusters containing points less than min_freq

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.
    """
    def __init__(self, radius=None, bandwidth=None, 
                 major_min_freq=None, 
                 minor_min_freq=None,
                 cluster_all=False):
        self.radius = radius
        self.bandwidth = bandwidth
        self.cluster_all = cluster_all
        self.major_min_freq = major_min_freq
        self.minor_min_freq = minor_min_freq

    def fit(self, X):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        y : Ignored
        """
        self.cluster_centers_, self.cluster_bandwidths_, self.labels_ = \
            modified_mean_shift(X, radius=self.radius, bandwidth=self.bandwidth, 
                                major_min_freq=self.major_min_freq, 
                                minor_min_freq=self.minor_min_freq,
                                cluster_all=self.cluster_all)
        
        # print(np.unique(self.labels_))
        # # print(self.cluster_centers_)
        # print(np.bincount(self.labels_+1))
        # plot_xy_with_label(X, self.labels_)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        labels, distances = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=METRIC, metric_kwargs=METRIC_PARAMS)

        if self.cluster_all is False:
            cluster_bandwidths_ = np.maximum(self.bandwidth, self.cluster_bandwidths_[labels])
            labels[distances > cluster_bandwidths_] = -1

        # print(np.unique(labels))
        # print(np.bincount(labels+1))
        # plot_xy_with_label(X, labels)

        return labels

    # def one_hot_predict(self, X):
    #     labels = self.predict(X).reshape(-1, 1)
    #     if self.cluster_all is True:
    #         return OneHotEncoder(n_values=self.n_cluster_).fit_transform(labels).toarray()
    #     else:
    #         return OneHotEncoder(n_values=self.n_cluster_).fit_transform(labels + 1).toarray()

    @property
    def n_cluster_(self):
        return len(self.cluster_centers_)

    @property
    def cluster_counts_(self):
        return np.bincount(self.labels_[self.labels_ != -1])
    #   valid_counts = self._original_cluster_counts[self._valid_labels]
    #   outlier_counts = np.sum(self._original_cluster_counts) - np.sum(valid_counts)
    #   return np.concatenate([outlier_counts, valid_counts], axis=0)


# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth,
                                       return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean
        my_mean = np.mean(points_within, axis=0)
        # If converged or at max_iter, adds the cluster
        if (np.linalg.norm(my_mean - my_old_mean) < stop_thresh or
                completed_iterations == max_iter):
            return tuple(my_mean), len(points_within)
        completed_iterations += 1


def modified_mean_shift(X, radius=None, bandwidth=None, 
                        major_min_freq=None, minor_min_freq=None,
                        cluster_all=True, max_iter=300):
    """Perform mean shift clustering of data using a flat kernel.

    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.
    labels : array, shape=[n_samples]
        Cluster labels for each point.
    """

    if radius is None or radius <= 0:
        raise ValueError("radius needs to be greater than zero and not None,\
            got %f" % radius)
    if bandwidth is None or bandwidth <= 0:
        raise ValueError("bandwidth needs to be greater than zero and not None,\
            got %f" % bandwidth)
    if major_min_freq is None or major_min_freq <= 0:
        raise ValueError("major_min_freq needs to be greater than zero and not None,\
            got %f" % major_min_freq)
    if minor_min_freq is None or minor_min_freq <= 0:
        raise ValueError("minor_min_freq needs to be greater than zero and not None,\
            got %f" % minor_min_freq)

    n_samples, n_features = X.shape

    # '-1' refers noises
    labels = np.zeros(n_samples, dtype=np.int)
    labels.fill(-1)

    # ###
    # # FIRST PART: NARROW AND FREQUENT CLUSTERS 
    # # (E.g., Home, office, ...)
    # ###

    # # scatter data in a search space
    # nbrs = NearestNeighbors(radius=radius, 
    #                         metric=METRIC, p=2, 
    #                         metric_params=METRIC_PARAMS).fit(X)
    # # allocating labels
    # for i, row in enumerate(X):
    #     # find neighboring data in the search space
    #     i_nbrs = nbrs.radius_neighbors([row], radius, return_distance=False)[0]
    #     # allocating new or existing labels
    #     existing_labels = sorted([labels[i] for i in i_nbrs if labels[i] != -1])
    #     if existing_labels:
    #         new_label = existing_labels[0] # set to the minimum label
    #     else:
    #         new_label = max(labels) + 1 # set to the new label
    #     labels[i_nbrs] = new_label

    # # remove imfrequent clusters
    # bincounts = np.bincount(labels)
    # label_map = dict()
    # for label, count in enumerate(bincounts):
    #     if count < major_min_freq:
    #         # remove
    #         labels[labels == label] = -1
    #     else:
    #         # prepare new label number
    #         if len(label_map) == 0:
    #             label_map[label] = 0
    #         else:
    #             label_map[label] = len(label_map)
    # # continual label numbers
    # for old_label, new_label in label_map.items():
    #     labels[labels == old_label] = new_label

    # SAVE cluster centers
    # if only cluster exist:
    if np.max(labels) + 1 > 0:
        cluster_centers = np.stack([np.mean(X[labels == label], axis=0) 
                                    for label in range(np.max(labels) + 1)])
    else:
        cluster_centers = None


    # # print(np.unique(labels[labels != -1]))
    # # # print(cluster_centers)
    # # plot_xy_with_label(X, labels)

    ###
    # SECOND PART: WIDE AND SOMEWHAT FREQUENT CLUSTERS
    # (E.g., near Home, near office, ...)
    ###

    # Target points of mean_shift clustering
    idxs_remain = (labels == -1)

    if np.sum(idxs_remain) > 0:
        X_remain = X[idxs_remain]

        # count bincounts
        center_intensity_dict = {}
        nbrs = NearestNeighbors(radius=bandwidth, 
                                metric=METRIC, p=2, 
                                metric_params=METRIC_PARAMS).fit(X_remain)

        # execute iterations on all seeds in parallel
        all_res = [_mean_shift_single_seed(row, X_remain, nbrs, max_iter) for row in X_remain]
        # copy results in a dictionary
        for i in range(len(X_remain)):
            if all_res[i] is not None:
                # {cluster_centroid: cluster_size}
                center_intensity_dict[all_res[i][0]] = all_res[i][1]

        # POST PROCESSING: remove near duplicate points
        # If the distance between two kernels is less than the bandwidth,
        # then we have to remove one because it is a duplicate. Remove the
        # one with fewer points.
        sorted_by_intensity = sorted(center_intensity_dict.items(),
                                    key=lambda tup: tup[1], reverse=True)
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        sorted_counts = np.array([tup[1] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=np.bool)
        nbrs = NearestNeighbors(radius=min_dist_between_clusters,
                                metric=METRIC, 
                                p=2, 
                                metric_params=METRIC_PARAMS).fit(sorted_centers)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center],
                                                    return_distance=False)[0]
                neighbor_count = np.sum(sorted_counts[neighbor_idxs])
                unique[neighbor_idxs] = 0
                unique[i] = 1
        cluster_centers_remain = sorted_centers[unique]

        # POST PROCESSING 2: remove imfrequent clusters
        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers_remain)
        distances, idxs = nbrs.kneighbors(X_remain)

        frequent_labels = np.bincount(idxs.flatten()) >= minor_min_freq
        cluster_centers_remain = cluster_centers_remain[frequent_labels]

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers_remain)
        distances, idxs = nbrs.kneighbors(X_remain)

        labels_remain = np.zeros((np.sum(idxs_remain), ), dtype=np.int)

        if cluster_all:
            labels_remain = idxs.flatten()
        else:
            labels_remain.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels_remain[bool_selector] = idxs.flatten()[bool_selector]

        # print(np.unique(labels_remain))
        # # print(cluster_centers_remain)

        labels_remain[labels_remain != -1] += np.max(labels) + 1
        labels[idxs_remain] = labels_remain
        if cluster_centers is not None:
            cluster_centers = np.concatenate([cluster_centers, cluster_centers_remain], axis=0)
        else:
            cluster_centers = cluster_centers_remain

    diatances_from_center = np.array([dist(row, cluster_centers[labels[i]], to_km=True)
                                        for i, row in enumerate(X)])
    # cluster_bandwidths = np.stack([np.median(diatances_from_center[labels == label], axis=0)
    #                                for label in range(np.max(labels) + 1)])
    cluster_bandwidths = np.stack([2.4 for label in range(np.max(labels) + 1)])

    return cluster_centers, cluster_bandwidths, labels