""" ========= METRIC FOR EVAL ========= """

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import weights_broadcast_ops

from custom_loss import compute_squared_distance_by_instance

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  """Creates a new local variable.
  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.
    validate_shape: Whether to validate the shape of the variable.
    dtype: Data type of the variables.
  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)


def _safe_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is <= 0.
  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.
  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)


def _mean(values, metrics_collections=None,
          updates_collections=None, name=None):
  """
  divide the total by counts, returns a mean value and a update_op
  """
  with variable_scope.variable_scope(name, 'mean', (values, )):
    values = math_ops.to_float(values)

    total = _create_local('total', shape=[])
    count = _create_local('count', shape=[])

    num_values = math_ops.to_float(array_ops.size(values))

    update_total_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)

    mean_t = _safe_div(total, count, 'value')
    update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_t, update_op


def _std(values, metrics_collections=None,
         updates_collections=None, name=None):
  """
  returns a std value and a update_op
  """
  with variable_scope.variable_scope(name, 'std', (values, )):
    values = math_ops.to_float(values)

    num_values = math_ops.to_float(array_ops.size(values))
    mean_value = _safe_div(math_ops.reduce_sum(values), num_values, 'mean')

    squared_deviation = math_ops.reduce_sum(math_ops.square(values - mean_value))

    total = _create_local('total', shape=[])
    count = _create_local('count', shape=[])

    update_total_op = state_ops.assign_add(total, math_ops.sqrt(num_values * squared_deviation))
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)

    std_t = _safe_div(total, count, 'value')
    update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, std_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return std_t, update_op


def _min(values, metrics_collections=None,
         updates_collections=None, name=None):
  """
  returns a min value and a update_op
  """
  with variable_scope.variable_scope(name, 'min', (values, )):
    values = math_ops.to_float(values)

    min_local = _create_local('min', shape=[])
    update_op = state_ops.assign_add(min_local, math_ops.reduce_min(values))
    min_t = array_ops.identity(min_local)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, min_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return min_t, update_op


def _max(values, metrics_collections=None,
         updates_collections=None, name=None):
  """
  returns a max value and a update_op
  """
  with variable_scope.variable_scope(name, 'max', (values, )):
    values = math_ops.to_float(values)

    max_local = _create_local('max', shape=[])
    update_op = state_ops.assign_add(max_local, math_ops.reduce_max(values))
    max_t = array_ops.identity(max_local)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, max_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return max_t, update_op


def _argmin(values, metrics_collections=None,
         updates_collections=None, name=None):
  """
  returns a argmin index and a update_op
  """
  with variable_scope.variable_scope(name, 'argmin', (values, )):
    values = math_ops.to_float(values)

    argmin_local = _create_local('argmin', shape=[], dtype=dtypes.int64)
    update_op = state_ops.assign_add(argmin_local, math_ops.argmin(values))
    argmin_t = array_ops.identity(argmin_local)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, argmin_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return argmin_t, update_op


def _argmax(values, metrics_collections=None,
         updates_collections=None, name=None):
  """
  returns a argmax index and a update_op
  """
  with variable_scope.variable_scope(name, 'argmax', (values, )):
    values = math_ops.to_float(values)

    argmax_local = _create_local('argmax', shape=[], dtype=dtypes.int64)
    update_op = state_ops.assign_add(argmax_local, math_ops.argmax(values))
    argmax_t = array_ops.identity(argmax_local)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, argmax_t)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return argmax_t, update_op


def mean_distance_metric(labels, predictions,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  """
  calculate the mean of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return _mean(distances, metrics_collections, updates_collections, name or 'mean_distance')


def std_distance_metric(labels, predictions,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """
  calculate the std of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return _std(distances, metrics_collections, updates_collections, name or 'std_distance')


def min_distance_metric(labels, predictions,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """
  calculate the min of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return _min(distances, metrics_collections, updates_collections, name or 'min_distance')


def max_distance_metric(labels, predictions,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """
  calculate the min of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return _max(distances, metrics_collections, updates_collections, name or 'max_distance')


def summary_statistics_metric(labels, predictions,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """
  calculate the summary_statistics_metric of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return (
      _mean(distances, metrics_collections, updates_collections, name or 'mean_distance'),
      _std(distances, metrics_collections, updates_collections, name or 'std_distance'),
      _min(distances, metrics_collections, updates_collections, name or 'min_distance'),
      _max(distances, metrics_collections, updates_collections, name or 'max_distance'),
      _argmin(distances, metrics_collections, updates_collections, name or 'argmin_index'),
      _argmax(distances, metrics_collections, updates_collections, name or 'argmax_index'),
  )