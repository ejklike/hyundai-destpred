""" ========= LOSS FOR TRAIN ========= """

from numpy import pi as pi_const
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util

def compute_squared_distance_by_instance(labels, predictions):
  """compute the squared distance by instance
  input: labels, predictions
  return: a tensor of size (batch_size, )
  """
  # 경도 (y): 1도= 88.8km, 1분=1.48km, 1초≒25.0m (위도 37도 기준)
  # 위도 (x): 1도=111.0Km, 1분=1.85Km, 1초=30.8m
  # http://lovestudycom.tistory.com/entry/위도-경도-계산법
  unit = 100 # to prevent NanLossDuringTrainingError
  km_per_latitude, km_per_longitude = 111.0/unit, 88.8/unit
  squared_delta = math_ops.squared_difference(predictions, labels)
  weights = ops.convert_to_tensor([[km_per_latitude**2, km_per_longitude**2], ], 
                                  dtype=squared_delta.dtype)
  weights = weights_broadcast_ops.broadcast_weights(
      math_ops.to_float(weights), squared_delta)
  squared_rescaled = math_ops.multiply(squared_delta, weights)
  sum_of_squared_rescaled = math_ops.reduce_sum(squared_rescaled, 1)
  return sum_of_squared_rescaled * unit**2


def compute_mean_loss(losses, scope=None, 
                      loss_collection=ops.GraphKeys.LOSSES):
  """Computes the mean loss.
  """
  with ops.name_scope(scope, "mean_loss", (losses, )):
    losses = ops.convert_to_tensor(losses)
    input_dtype = losses.dtype
    losses = math_ops.to_float(losses)
    loss = math_ops.reduce_mean(losses)

    # Convert the result back to the input type.
    loss = math_ops.cast(loss, input_dtype)
    util.add_loss(loss, loss_collection)
    return loss


def mean_squared_distance_loss(labels, predictions, scope=None,
                               add_collection=False):
  """
  Adds a Mean-squared-distance loss to the training procedure.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "squared_distance_loss",
                      (predictions, labels)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    squared_distances = compute_squared_distance_by_instance(predictions, labels)

    loss_collection = ops.GraphKeys.LOSSES if add_collection else None
    return compute_mean_loss(squared_distances, scope, loss_collection=loss_collection)


def neg_log_likelihood_loss(labels, pi, m1, m2, s1, s2, rho, scope=None,
                            add_collection=False):
  """
  Adds a Mean neg_log_likelihood loss to the training procedure.
   : P(labels | params returned by model)
  """
  with ops.name_scope(scope, "neg_lig_likelihood_loss",
                      (labels, pi, m1, m2, s1, s2, rho, )) as scope:

    nll_losses = compute_neg_log_likelihood_by_instance(labels, pi, m1, m2, s1, s2, rho)

    loss_collection = ops.GraphKeys.LOSSES if add_collection else None
    return compute_mean_loss(nll_losses, scope, loss_collection=loss_collection)


def compute_neg_log_likelihood_by_instance(labels, pi, m1, m2, s1, s2, rho):
  """
  return neg_log_likelihood
  """
  # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
  x1, x2 = array_ops.split(labels, num_or_size_splits=2, axis=1, name='label_split')
  # subtract DOES SUPPORT broadcasting!
  n1_s1 = math_ops.div(math_ops.subtract(x1, m1), s1)
  n2_s2 = math_ops.div(math_ops.subtract(x2, m2), s2)
  neg_rho = math_ops.subtract(1., math_ops.square(rho))
  
  z = math_ops.square(n1_s1) + math_ops.square(n2_s2)
  z -= 2. * math_ops.multiply(rho, math_ops.multiply(n1_s1, n2_s2))
  denom = 2. * pi_const * math_ops.multiply(math_ops.multiply(s1, s2), math_ops.sqrt(neg_rho))
  
  normal = math_ops.div(math_ops.exp(math_ops.div(-z, 2 * neg_rho)), denom)

  # implementing eq # 26 of http://arxiv.org/abs/1308.0850
  epsilon = 1e-20
  mixture_normal =math_ops.reduce_sum(math_ops.multiply(pi, normal), axis=1)
  ops.add_to_collection('m_normal', mixture_normal)
  return -math_ops.log(math_ops.maximum(mixture_normal, epsilon))