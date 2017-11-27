import numpy as np

import tensorflow as tf

from custom_loss import mean_squared_distance_loss
from custom_metric import summary_statistics_metric


def length(sequence):
  """return sequenc-by-sequence lengths
  input: sequence of shape [batch_size, seq_length, 3]
  output: sequence lengths [batch_size,]
  """
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2)) # [batch_size, seq_length]
  length = tf.reduce_sum(used, 1) # [batch_size,]
  return tf.cast(length, tf.int32)


def predict_next(input_path, cell, state_in, params):

  # DECODER output
  with tf.variable_scope('rnn_part') as scope:
    outputs, state_out = tf.nn.dynamic_rnn(cell, input_path, 
                                           initial_state=state_in, 
                                           sequence_length=length(input_path),
                                           scope='rnn')
  n_out = 2
  with tf.variable_scope('linear_part') as scope:
    output_w = tf.get_variable("kernel", [params['rnn_size'], n_out])
    output_b = tf.get_variable("bias", [n_out])
    # [batch, seq_length, rnn_size] ==> [batch * seq_len, rnn_size]
    output = tf.reshape(outputs, (-1, params['rnn_size']), name='flat_output')
    # [batch * seq_len, rnn_size] * [rnn_size, n_out] = [batch * seq_len, n_out]
    output = tf.nn.xw_plus_b(output, output_w, output_b, name='flat_output_point')
  return outputs, state_out


def loss_and_train_op(target_path, predicted_path, learning_rate=None):
  """LOSS and TRAIN_OP"""
  flat_target_path = tf.reshape(target_path, [-1, 2])
  flat_predicted_path = tf.reshape(predicted_path, [-1, 2])

  mask = tf.cast(tf.sign(tf.reduce_sum(tf.abs(flat_target_path), 1)), dtype=tf.bool)

  flat_target_path = tf.boolean_mask(flat_target_path, mask)
  flat_predicted_path = tf.boolean_mask(flat_predicted_path, mask)

  dist_loss = mean_squared_distance_loss(flat_target_path, flat_predicted_path)
  tf.add_to_collection('losses', dist_loss)

  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  if learning_rate is None:
    return loss
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return loss, train_op


# def predict(pi, mu1, mu2, sig1, sig2, rho, eop, params):
#   """
#   # WARNING: THIS CODE ONLY WORKS WITH BATCH SIZE 1 !!!
#   # batch_size MUST BE "1" for valid prediction!!!!
#   """
#   eop = eop[-1]
#   param_list = [p[-1] for p in [pi, mu1, mu2, sig1, sig2, rho]]

#   def _get_mask_value(param, argmax_mask):
#     return tf.reduce_sum(tf.multiply(param, argmax_mask), 
#                          name='%s_of_argmax' % param.name[:-2])

#   argmax_mask = tf.one_hot(tf.argmax(pi, axis=1), depth=params['n_mixture'])
#   pi, mu1, mu2, sig1, sig2, rho = [_get_mask_value(p, argmax_mask) for p in param_list]
  
#   mu = tf.stack([mu1, mu2], axis=0, name='mu')
#   cov = tf.concat([tf.reshape(tf.stack([sig1*sig1, rho*sig1*sig2], axis=0), (-1, 1)), 
#                   tf.reshape(tf.stack([rho*sig1*sig2, sig2*sig2], axis=0), (-1, 1))], 
#                   axis=1, name='cov')
#   return mu, cov, eop