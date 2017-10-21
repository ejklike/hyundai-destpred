import tensorflow as tf
from tensorflow.contrib import rnn

def _variable_on_cpu(name, shape):
  initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def rnn_last_output(rnn_input, n_unit=16, bi_direction=False):
  rnn_cell = rnn.BasicLSTMCell(n_unit)
  if bi_direction:
    outputs, _, _ = rnn.static_bidirectional_rnn(
            rnn_cell, rnn_cell, rnn_input, dtype=tf.float32)
  else:
    outputs, _ = rnn.static_rnn(
            rnn_cell, rnn_input, dtype=tf.float32)
  return outputs[-1]


def embed_path(paths, params):
  if params['model_type'] == 'dnn':
    nn_inputs = tf.reshape(paths, shape=[-1, 4 * params['k']]) ###?
    return tf.layers.dense(nn_inputs, 
                           params['path_embedding_dim'], 
                           activation=tf.nn.relu)
  else:
    rnn_inputs = tf.unstack(paths, axis=1, name='unstack')
    return rnn_last_output(rnn_inputs, 
                           n_unit=params['path_embedding_dim'], 
                           bi_direction=params['bi_direction'])


def embed_meta(metas, params):
  holiday, weekno, hour, weekday = tf.split(metas, metas.shape[1], axis=1)
  
  holi_W = _variable_on_cpu('holiday_table', [2, 1])
  holiday_lookup = tf.nn.embedding_lookup(holi_W, holiday, name='holiday_lookup')
  holiday_embedding = tf.squeeze(holiday_lookup, axis=1, name='holiday_embedding')

  # weekno_W = _variable_on_cpu('weekno_table', [53, 10])
  # weekno_lookup = tf.nn.embedding_lookup(weekno_W, weekno, name='weekno_lookup')
  # weekno_embedding = tf.squeeze(weekno_lookup, axis=1, name='weekno_embedding')

  hour_W = _variable_on_cpu('hour_table', [24, 4])
  hour_lookup = tf.nn.embedding_lookup(hour_W, hour, name='hour_lookup')
  hour_embedding = tf.squeeze(hour_lookup, axis=1, name='hour_embedding')

  weekday_W = _variable_on_cpu('weekday_table', [7, 2])
  weekday_lookup = tf.nn.embedding_lookup(weekday_W, weekday, name='weekday_lookup')
  weekday_embedding = tf.squeeze(weekday_lookup, axis=1, name='weekday_embedding')

  return tf.concat([holiday_embedding, 
                    # weekno_embedding, 
                    hour_embedding, 
                    weekday_embedding], axis=1)


def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  paths, metas = features['path'], features['meta']
  paths = tf.cast(paths, dtype=tf.float32)

  # PATH embedding
  path_embedding = embed_path(paths, params) if params['use_path'] else None
  # META embedding
  meta_embedding = embed_meta(metas, params) if params['use_meta'] else None

  # CONCAT
  key = ''.join(['meta' if params['use_meta'] else '',
                 'path' if params['use_path'] else ''])
  concat_target = dict(
    path=[path_embedding],
    meta=[meta_embedding],
    metapath=[path_embedding, meta_embedding])[key]
  x = tf.concat(concat_target, axis=1, name='embedded_input')
  
  # FINAL dense layer
  for i_layer in range(1, params['n_hidden_layer'] + 1):
    n_hidden_node = x.get_shape().as_list()[1]
    x = tf.layers.dense(x, n_hidden_node, activation=tf.nn.relu)
  predictions = tf.layers.dense(x, 2, activation=None)

  if mode == tf.estimator.ModeKeys.TRAIN:
    print('Trainable Variables')
    for v in tf.trainable_variables():
      print(v.name + ':' + str(v.shape))

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # Define loss, optimizer, and train_op
  labels = tf.cast(labels, dtype=tf.float32)
  loss = tf.losses.mean_squared_error(labels, predictions)
  optimizer = tf.train.AdamOptimizer(
                                learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(loss=loss, 
                                global_step=tf.train.get_global_step())

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = dict(
      rmse=tf.metrics.root_mean_squared_error(labels, predictions)
  )

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)
