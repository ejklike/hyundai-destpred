import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, shape):
  initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def length(sequence):
  """return sequenc-by-sequence lengths
  input: sequence of shape [batch_size, seq_length, None]
  output: sequence lengths [batch_size,]
  """
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2)) # [batch_size, seq_length]
  length = tf.reduce_sum(used, 1) # [batch_size,]
  return tf.cast(length, tf.int32)


def rnn_last_output(rnn_input, n_unit=16, bi_direction=False, scope=None):
  """
  input: [batch_size, time_size, 2]
  output: [batch_size, n_unit] if bi_direction is False else [batch_size, n_unit * 2]
  """
  rnn_cell = rnn.BasicLSTMCell(n_unit)
  if bi_direction:
      outputs, _, = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell, rnn_cell, rnn_input, dtype=tf.float32, scope=scope, 
          parallel_iterations=100,
          sequence_length=length(rnn_input))
      outputs = tf.concat(outputs, axis=2) # concat fw and bw outputs
  else:
      outputs, _ = tf.nn.dynamic_rnn(
          rnn_cell, rnn_input, dtype=tf.float32, scope=scope, 
          parallel_iterations=100,
          sequence_length=length(rnn_input))
  return tf.unstack(outputs, axis=1, name='unstack')[-1]


def embed_path(paths):
  # print('path embedded')
  if FLAGS.model_type == 'dnn':
    # paths = tf.reshape(paths, # first and last k points for both x and y
    #                        shape=[-1, 2 * 2 * FLAGS.k])
    return tf.layers.dense(paths, 
                           FLAGS.path_embedding_dim, 
                           activation=tf.nn.relu, 
                           name='embed_path')
  else:
    return rnn_last_output(paths, 
                           n_unit=FLAGS.path_embedding_dim, 
                           bi_direction=FLAGS.bi_direction,
                           scope='embed_path')


def embed_meta(metas):
  # print('meta embedded')
  holiday, weekno, hour, weekday = tf.split(metas, metas.shape[1], axis=1)
  
  with tf.variable_scope('embed_meta') as scope:

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


def extract_feature(path_t, meta_t):
  """ return final prediction results
  """
  # Regularization parameters
  keep_prob = FLAGS.keep_prob
  reg_scale = FLAGS.reg_scale
  # dense layer parameters
  n_hidden_layer = FLAGS.n_hidden_layer
  n_hidden_node = FLAGS.n_hidden_node

  # Embed path and meta data
  embedding_results = []

  if FLAGS.use_meta:
    meta_embedding = embed_meta(meta_t)
    embedding_results.append(meta_embedding)

  if FLAGS.use_path:
    path_embedding = embed_path(path_t)
    embedding_results.append(path_embedding)

  # Concat embeddings to feed to classification/regression part
  x = tf.concat(embedding_results, axis=1, name='concat_embedded_input')
  x = tf.nn.dropout(x, keep_prob=keep_prob, name='concat_dropout')

  # Stack dense layers (#: n_hidden_layer)
  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
  for i_layer in range(1, n_hidden_layer + 1):
    x = tf.layers.dense(x, n_hidden_node, 
                        activation=tf.nn.relu, 
                        name='dense_%d'%i_layer, 
                        kernel_regularizer=kernel_regularizer)
    x = tf.nn.dropout(x, 
                      keep_prob=keep_prob, 
                      name='dense_%d_dropout'%i_layer)

  return x


