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


def mean(values, metrics_collections=None,
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

  
  return 


def mean_distance_metric(labels, predictions,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  """
  calculate the mean of distances in kilometer unit
  """
  squared_deistances = compute_squared_distance_by_instance(labels, predictions)
  distances = math_ops.sqrt(squared_deistances)

  return mean(distances, metrics_collections,
              updates_collections, name or 'mean_distance')