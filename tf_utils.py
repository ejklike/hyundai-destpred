import numpy as np
import tensorflow as tf
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import MeanShift


class ModifiedMeanShift(object):
  """
  exclude clusters containing points less than min_freq
  """

  def __init__(self, bandwidth=0.01, min_freq=3):
    self.mean_shift = MeanShift(bandwidth=bandwidth)
    self.min_freq = min_freq

  def fit(self, data):
    cluster_labels = self.mean_shift.fit_predict(data)

    self._original_cluster_counts = np.bincount(cluster_labels)
    self._original_centroids = self.mean_shift.cluster_centers_

    # map between new label and old label in fit results
    # (key, value: old, new)
    self._map = dict()
    self.max_new_label = 0
    for old_label, count in enumerate(self._original_cluster_counts):
      if count >= self.min_freq:
        self.max_new_label += 1
        self._map[old_label] = self.max_new_label
      else:
        self._map[old_label] = 0

    self._valid_labels = [old for old, new in self._map.items() if new > 0]

    print(self._original_cluster_counts)
    print(self._valid_labels)
    return self

  def predict(self, data):
    old_labels = self.mean_shift.predict(data)
    new_labels = np.array([self._map[l] for l in old_labels])
    return new_labels

  @property
  def n_cluster_(self):
    return len(self._valid_labels)

  # @property
  # def cluster_counts_(self):
  #   valid_counts = self._original_cluster_counts[self._valid_labels]
  #   outlier_counts = np.sum(self._original_cluster_counts) - np.sum(valid_counts)
  #   return np.concatenate([outlier_counts, valid_counts], axis=0)

  @property
  def cluster_centers_(self):
    valid_centers = self._original_centroids[self._valid_labels, :]
    number_of_outliers = len(self._original_centroids) - len(valid_centers)
    center_of_outliers = np.subtract(
        np.sum(self._original_centroids, axis=0), 
        np.sum(valid_centers, axis=0)
    ) / number_of_outliers
    return np.concatenate([center_of_outliers.reshape(1, 2), valid_centers], axis=0)


class NeighborWeightCalculator(object):

  def __init__(self, radius=5, reference_points=None):
    assert reference_points is not None
    # radius neighbor
    self.radius = radius
    # reference points to be counted
    self.ref_points = reference_points
    self.num_ref = len(reference_points)

  def get_neighbor_weight(self, target_points):
    """ return a weight proportion to the number of neighbors
    input: the centers of radius_neighbors
    output: a weight (sum to 1)
    """
    if self.radius > 0:
      #         trn     tst
      # trn | trn_trn trn_tst |
      # tst | tst_trn tst_tst |
      dest_all = np.concatenate([self.ref_points, target_points], axis=0)
      connectivity_matrix = radius_neighbors_graph(dest_all, self.radius, 
                                                  mode='connectivity',
                                                  include_self=True, # contain myself
                                                  # weighted distance
                                                  metric='wminkowski', 
                                                  metric_params={'w': [88.8**2, 111.0**2]},
                                                  p=2).toarray()
      # get only [trn_tst] part and apply reduce_sum to count all neighbors
      counts = np.sum(connectivity_matrix[:self.num_ref, self.num_ref:], axis=0)
    else:
      counts = np.ones([target_points.shape[0],], dtype=np.int32)
    return counts / np.sum(counts)


class BatchGenerator(object):

  def __init__(self, data_list, batch_size, epoch=None):
    self.pointer = 0
    self.counter = 0
    self.batch_size = batch_size
    self.epoch = epoch
    self.data_size = data_list[0].shape[0]
    self.data_list = [np.concatenate([data, data], axis=0) for data in data_list]

  def next_batch(self):
    if (self.epoch is not None) and (self.counter > self.data_size * self.epoch):
      raise tf.errors.OutOfRangeError('OutOfRange ERROR!')

    next_pointer = self.pointer + self.batch_size

    batch_list = [data[self.pointer:next_pointer] for data in self.data_list]
    self.counter += (next_pointer - self.pointer)
    self.pointer = (next_pointer) % self.data_size

    return batch_list


def compute_km_distances(dest1, dest2, name=None):
  """compute the squared distance by instance
  input: two destination tensors of size (batch_size, )
  return: a tensor of size (batch_size, )
  """
  # assert two destination tensors have compatible shapes
  dest1.get_shape().assert_is_compatible_with(dest2.get_shape())

  # http://lovestudycom.tistory.com/entry/위도-경도-계산법
  # 경도 (y): 1도= 88.8km, 1분=1.48km, 1초≒25.0m (위도 37도 기준)
  # 위도 (x): 1도=111.0Km, 1분=1.85Km, 1초=30.8m
  unit = 100
  km_per_latitude, km_per_longitude = 111.0/unit, 88.8/unit
  squared_weight = [km_per_latitude**2, km_per_longitude**2]

  with tf.name_scope(name, "compute_distances", [dest1, dest2]) as scope:
    dest1 = tf.convert_to_tensor(dest1, name="dest1")
    dest2 = tf.convert_to_tensor(dest2, name="dest2")
    # compute the squared distances weighted by km/degree
    squared_delta = tf.squared_difference(dest1, dest2)
    weighted_squared_delta = tf.multiply(squared_delta, squared_weight)
    sum_of_weighted_squared_delta = tf.reduce_sum(weighted_squared_delta, axis=1)
    distances = tf.sqrt(sum_of_weighted_squared_delta)

    return distances * unit
