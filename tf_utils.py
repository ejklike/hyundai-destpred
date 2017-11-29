import numpy as np
import tensorflow as tf
from sklearn.neighbors import radius_neighbors_graph


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
