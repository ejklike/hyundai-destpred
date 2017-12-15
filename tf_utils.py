import numpy as np
import tensorflow as tf


def trn_batch_generator(paths, metas, labels, batch_size=300, epoch=10000):

  label_sizes = np.bincount(labels)
  n_labels = len(label_sizes)
  label_idxs = [np.where(labels==i)[0] for i in range(n_labels)]

  batch_sizes = [label_sizes[i] if label_sizes[i] > 5 else 5 for i in range(n_labels)]
  batch_sizes[0] += batch_size - np.sum(batch_sizes)

  for i in range(epoch):

    batch_paths_list = []
    batch_metas_list = []
    batch_labels_list = []

    for j in range(n_labels):
      # print(label_idxs[j], batch_sizes[j])
      if len(label_idxs[j]) > 0:
        this_label_idxs = np.random.choice(label_idxs[j], batch_sizes[j])
        batch_paths_list.append(paths[this_label_idxs])
        batch_metas_list.append(metas[this_label_idxs])
        batch_labels_list.append(labels[this_label_idxs])

    batch_paths = np.concatenate(batch_paths_list, axis=0)
    batch_metas = np.concatenate(batch_metas_list, axis=0)
    batch_labels = np.concatenate(batch_labels_list, axis=0)

    yield batch_paths, batch_metas, batch_labels


class BatchGenerator(object):

  def __init__(self, data_list, batch_size=1000, epoch=10000):
    paths, metas, labels = data_list
    self.batch_generator = trn_batch_generator(paths, metas, labels, 
                                               batch_size=batch_size, epoch=epoch)

  def next_batch(self):
    return next(self.batch_generator)


# class BatchGenerator(object):

#   def __init__(self, data_list, batch_size, epoch=None):
#     self.data_list = data_list

#   def next_batch(self):
#     return self.data_list


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