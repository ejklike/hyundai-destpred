import tensorflow as tf
from tensorflow.contrib import rnn

from custom_loss import mean_squared_distance_loss, neg_log_likelihood_loss
from custom_metric import summary_statistics_metric


def _variable_on_cpu(name, shape):
  initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def rnn_last_output(rnn_input, n_unit=16, bi_direction=False, scope=None):
  """
  input: [batch_size, time_size, 2]
  output: [batch_size, n_unit] if bi_direction is False else [batch_size, n_unit * 2]
  """
  rnn_cell = rnn.BasicLSTMCell(n_unit)
  if bi_direction:
      outputs, _, = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell, rnn_cell, rnn_input, dtype=tf.float32, scope=scope)
      outputs = tf.concat(outputs, axis=2) # concat fw and bw outputs
  else:
      outputs, _ = tf.nn.dynamic_rnn(
          rnn_cell, rnn_input, dtype=tf.float32, scope=scope)
  return tf.unstack(outputs, axis=1, name='unstack')[-1]


def embed_path(paths, params):
  # print('path embedded')
  if params['model_type'] == 'dnn':
    nn_inputs = tf.reshape(paths, # first and last k points for both x and y
                           shape=[-1, 4 * params['k']])
    return tf.layers.dense(nn_inputs, 
                           params['path_embedding_dim'], 
                           activation=tf.nn.relu, name='embed_path')
  else:
    return rnn_last_output(paths, 
                           n_unit=params['path_embedding_dim'], 
                           bi_direction=params['bi_direction'],
                           scope='embed_path')


def embed_meta(metas, params):
  # print('meta embedded')
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


def build_graph(features, params):
  concat_target = []

  # META embedding
  if params['use_meta']:
    metas = features['meta']
    meta_embedding = embed_meta(metas, params)
    concat_target.append(meta_embedding)

  # PATH embedding  
  if params['use_path']:
    paths = features['path']
    paths = tf.cast(paths, dtype=tf.float32)
    path_embedding = embed_path(paths, params)
    concat_target.append(path_embedding)

  x = tf.concat(concat_target, axis=1, name='concat_embedded_input')

  # before FINAL dense layer
  for i_layer in range(1, params['n_hidden_layer'] + 1):
    # n_hidden_node = x.get_shape().as_list()[1]
    n_hidden_node = params['path_embedding_dim']
    x = tf.layers.dense(x, n_hidden_node, activation=tf.nn.relu, name='dense_%d'%i_layer)
  
  # FINAL prediction
  if params['cluster_bw'] > 0:
    c_centers = tf.constant(params['cluster_centers'], 
                            dtype=tf.float32, name='cluster_centers')
    c_probs = tf.layers.dense(x, params['n_clusters'], 
                              activation=tf.nn.softmax, name='cluster_probs')
    predictions = tf.nn.xw_plus_b(c_probs, c_centers, [0., 0.], name='final')
  else:
    predictions = tf.layers.dense(x, 2, activation=None, name='fianl')

  return predictions


def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # load validation data as constants
  features_val = params['features_val']
  labels_val = tf.constant(params['labels_val'], dtype=tf.float32)
  for k, v in features_val.items():
    if k == 'meta':
      features_val[k] = tf.constant(v, dtype=tf.int32)
    else:
      features_val[k] = tf.constant(v, dtype=tf.float32)

  # build graph for training or evaluation
  with tf.variable_scope('graph') as scope:
    # prediction for training data
    predictions = build_graph(features, params)
    scope.reuse_variables()
    # prediction for validation data
    predictions_val = build_graph(features_val, params)

  if mode == tf.estimator.ModeKeys.TRAIN:
    print('>>> Trainable Variables: ')
    for v in tf.trainable_variables():
      print(v.name + '; ' + str(v.shape))

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # Define loss, optimizer, and train_op
  labels = tf.cast(labels, dtype=tf.float32)
  loss = mean_squared_distance_loss(labels, predictions)
  optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
  # optimizer = tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"], momentum=0.9)
  # optimizer = tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], 
                                        #  momentum=0.95)
  train_op = optimizer.minimize(loss=loss, 
                                global_step=tf.train.get_global_step())

  # LOSS for validation
  loss_val = mean_squared_distance_loss(labels_val, predictions_val, scope='loss_val')
  
  _mean, _std, _min, _max, _argmin, _argmax = summary_statistics_metric(labels, predictions)

  # evaluation metric
  eval_metric_ops = dict(
      mean_distance=_mean,
      std_distance=_std,
      min_distance=_min,
      max_distance=_max,
      argmin_index=_argmin,
      argmax_index=_argmax,
  )

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def build_mdn_graph(features, params):
  concat_target = []

  # META embedding
  if params['use_meta']:
    metas = features['meta']
    meta_embedding = embed_meta(metas, params)
    concat_target.append(meta_embedding)

  # PATH embedding  
  if params['use_path']:
    paths = features['path']
    paths = tf.cast(paths, dtype=tf.float32)
    path_embedding = embed_path(paths, params)
    concat_target.append(path_embedding)

  x = tf.concat(concat_target, axis=1, name='concat_embedded_input')

  # before FINAL dense layer
  for i_layer in range(1, params['n_hidden_layer'] + 1):
    # n_hidden_node = x.get_shape().as_list()[1]
    n_hidden_node = params['path_embedding_dim']
    x = tf.layers.dense(x, n_hidden_node, activation=tf.nn.relu, name='dense_%d'%i_layer)

  # param prediction
  predictions = tf.layers.dense(x, 6 * params['n_mixture'], activation=None, name='generate_params')
  pi, mu1, mu2, sig1, sig2, rho = tf.split(predictions, axis=1, num_or_size_splits=6, 
                                           name='split_params')
  pi = tf.nn.softmax(pi, name='pi_softmax')
  sig1, sig2 = tf.exp(sig1, name='sig1_exp'), tf.exp(sig2, name='sig2_exp')
  rho = tf.tanh(rho, name='rho_tanh')
  return pi, mu1, mu2, sig1, sig2, rho


def model_fn_mdn(features, labels, mode, params):
  """MDN Model function for Estimator."""

  # load validation data as constants
  features_val = params['features_val']
  labels_val = tf.constant(params['labels_val'], dtype=tf.float32)
  for k, v in features_val.items():
    if k == 'meta':
      features_val[k] = tf.constant(v, dtype=tf.int32)
    else:
      features_val[k] = tf.constant(v, dtype=tf.float32)

  # build graph for training or evaluation
  with tf.variable_scope('graph') as scope:
    # prediction for training data
    pi, mu1, mu2, sig1, sig2, rho = build_mdn_graph(features, params)
    scope.reuse_variables()
    # prediction for validation data
    pi_v, mu1_v, mu2_v, sig1_v, sig2_v, rho_v = build_mdn_graph(features_val, params)

    # Predictions for evaluation
    argmax_mask = tf.one_hot(tf.argmax(pi, axis=1), depth=params['n_mixture'])
    
    mu1_max = tf.reduce_sum(tf.multiply(mu1, argmax_mask), axis=1, keep_dims=True)
    mu2_max = tf.reduce_sum(tf.multiply(mu2, argmax_mask), axis=1, keep_dims=True)
    predictions = tf.concat([mu1_max, mu2_max], axis=1)

  if mode == tf.estimator.ModeKeys.TRAIN:
    print('>>> Trainable Variables: ')
    for v in tf.trainable_variables():
      print(v.name + '; ' + str(v.shape))

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  # Define loss, optimizer, and train_op
  labels = tf.cast(labels, dtype=tf.float32)
  loss = neg_log_likelihood_loss(labels, pi, mu1, mu2, sig1, sig2, rho)
  optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(loss=loss, 
                                global_step=tf.train.get_global_step())

  # LOSS for validation
  loss_val = neg_log_likelihood_loss(labels_val, pi_v, mu1_v, mu2_v, sig1_v, sig2_v, rho_v, 
                                     scope='loss_val')

  _mean, _std, _min, _max = summary_statistics_metric(labels, predictions)

  # evaluation metric
  eval_metric_ops = dict(
      mean_distance=_mean,
      std_distance=_std,
      min_distance=_min,
      max_distance=_max,
      argmin_index=_argmin,
      argmax_index=_argmax,
  )

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)
