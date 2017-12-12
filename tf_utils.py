import numpy as np
import tensorflow as tf


class BatchGenerator(object):

  def __init__(self, data_list):
    self.data_list = data_list

  def next_batch(self):
    return self.data_list


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
