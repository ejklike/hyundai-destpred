import os
import time

import numpy as np
np.random.seed(42)

import tensorflow as tf
from tensorflow.contrib import rnn

from log import log
from utils import maybe_exist
from tf_utils import (BatchGenerator, 
                      compute_km_distances)
from graph import (extract_feature, length)

# FLAGS to be applied across related modules
FLAGS = tf.flags.FLAGS


def encode_axis(x, keep_prob, scope=None):
  if FLAGS.model_type == 'rnn':
    rnn_cell = rnn.BasicLSTMCell(FLAGS.num_units) # FLAGS.path_embedding_dim
    # rnn_cell = rnn.MultiRNNCell([rnn_cell] * 2)
    rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
    if FLAGS.bi_direction:
      outputs, _, = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell, rnn_cell, x, dtype=tf.float32, scope=scope, parallel_iterations=100)
      outputs = tf.concat(outputs, axis=2) # concat fw and bw outputs
    else:
      outputs, _ = tf.nn.dynamic_rnn(
          rnn_cell, x, dtype=tf.float32, scope=scope, parallel_iterations=1000)
      
    x = tf.unstack(outputs, axis=1, name='unstack')[-1]

    # feature_size = FLAGS.seq_len * FLAGS.num_units * (2 if FLAGS.bi_direction else 1)
    # x = tf.reshape(outputs, (-1, feature_size))

  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale)
  for i_layer in range(1, FLAGS.n_hidden_layer + 1):
    x = tf.layers.dense(x, FLAGS.n_hidden_node, 
                        activation=tf.nn.relu, 
                        name=scope + '_' + 'dense_%d'%i_layer, 
                        kernel_regularizer=kernel_regularizer)
    x = tf.nn.dropout(x, 
                      keep_prob=keep_prob, 
                      name=scope + '_' + 'dense_%d_dropout'%i_layer)
  return tf.layers.dense(x, 1, activation=None, name=scope + '_' +  'fianl')


class Model(object):

  def __init__(self, model_dir):
    # Get model_dir
    maybe_exist(model_dir)
    self.model_dir = model_dir
    log.info('model_dir: %s', model_dir)


  def build_graph(self):
    # Parameters
    learning_rate = FLAGS.learning_rate

    # Reset default_graph
    tf.reset_default_graph()

    # with tf.Graph().as_default():
    # Define global_step
    self.global_step = tf.train.get_or_create_global_step()

    # Placeholders
    self.path_t = tf.placeholder(dtype=tf.float32, 
                                 shape=[None, FLAGS.seq_len, 2],
                                 name='path_placeholder')
    self.meta_t = tf.placeholder(dtype=tf.int32, 
                                 shape=[None, 4], 
                                 name='meta_placeholder')
    self.dest_t = tf.placeholder(dtype=tf.float32, 
                                 shape=[None, 2], 
                                 name='dest_placeholder')
    self.keep_t = tf.placeholder(dtype=tf.float32, 
                                 shape=None, 
                                 name='keep_placeholder')
    
    if FLAGS.model_type == 'dnn':
      path_x, path_y = tf.unstack(self.path_t, axis=2)
    else:
      path_x, path_y = tf.split(self.path_t, axis=2, num_or_size_splits=2)

    x_axis = encode_axis(path_x, self.keep_t, scope='encode_x')
    y_axis = encode_axis(path_y, self.keep_t, scope='encode_y')

    self.pred_t = tf.concat([x_axis, y_axis], axis=1)

    # Define loss
    self.distances = compute_km_distances(self.dest_t, self.pred_t)
    self.average_loss_t = tf.reduce_mean(self.distances, name='average_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, self.average_loss_t)


    # Gather all losses
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss_t = tf.add_n(losses, name='training_loss')

    # Training Op.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = optimizer.minimize(loss=self.loss_t, 
                                       global_step=tf.train.get_global_step())

    # Initializing Op.
    self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a saver for writing training checkpoints.
    self.saver = tf.train.Saver(max_to_keep=1)

    # Create a session for running Ops on the Graph.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
    self.sess = tf.Session(config=sess_config)


  def init_or_restore_all_variables(self, restart=False):
    # Remove prev model or not
    if restart is True and self.latest_checkpoint is not None:
      log.warning('Delete prev model_dir: %s', self.model_dir)
      tf.gfile.DeleteRecursively(self.model_dir)
    
    # latest_checkpoint
    latest_ckpt_path = self.latest_checkpoint
    # Initialize OR restore variables
    if latest_ckpt_path is None:
      self._init_all_trainable_variables()
    else:
      self._restore_all_trainable_variables(latest_ckpt_path)


  @property
  def latest_step(self):
    """return the latest step"""
    return self.sess.run(self.global_step)

  @property
  def latest_checkpoint(self):
    """return the latest checkpoint path"""
    return tf.train.latest_checkpoint(self.model_dir)


  def _save_ckpt(self, step):
    """Saves the latest checkpoint."""
    log.info("...Saving checkpoints for %d.", step)
    save_path = os.path.join(self.model_dir, 'model.ckpt')
    self.saver.save(self.sess, save_path, global_step=step)


  def _init_all_trainable_variables(self):
    """Create and run the Op to initialize the variables."""
    log.info('Initialize model parameters...')
    self.sess.run(self.init_op)


  def _restore_all_trainable_variables(self, ckpt_path):
    """Restore variable values from the given checkpoint path"""
    log.info('Restore from the previous ckpt path: %s', ckpt_path)
    self.saver.restore(self.sess, ckpt_path)


  def print_all_trainable_variables(self):
    print('>>> Trainable Variables: ')
    for v in tf.trainable_variables():
      print('   ', v.name + '; ' + str(v.shape))



  def train(self, path, meta, dest):
    """ train the model using the input training data
    """

    # Parameters for training
    early_stopping_rounds = FLAGS.early_stopping_rounds
    log_freq = FLAGS.log_freq

    # Split train set into train/validation sets
    num_data = len(path)
    num_trn = int(num_data * (1 - FLAGS.validation_size))
    trn_idx = np.random.choice(num_data, num_trn, replace=False)
    mask = np.zeros((num_data, ), dtype=bool)
    mask[trn_idx] = True
    path_trn, path_val = path[mask], path[~mask]
    meta_trn, meta_val = meta[mask], meta[~mask]
    dest_trn, dest_val = dest[mask], dest[~mask]
    # path_trn, path_val = path[:num_trn], path[num_trn:]
    # meta_trn, meta_val = meta[:num_trn], meta[num_trn:]
    # dest_trn, dest_val = dest[:num_trn], dest[num_trn:]

    # Instantiate batch generator
    trn_batch_generator = BatchGenerator([path_trn, meta_trn, dest_trn])

    # Define feed_dict for validation and early stopping
    feed_dict_val = {self.path_t: path_val, 
                     self.meta_t: meta_val, 
                     self.dest_t: dest_val,
                     self.keep_t: 1}

    # And then after everything is built, start the training loop.
    try:
      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      
      # Best loss and its step
      best_loss, best_step = None, 0
      while not coord.should_stop():

        start_time = time.time()

        # Run one step of the model.
        this_path, this_meta, this_dest = trn_batch_generator.next_batch()
        feed_dict = {self.path_t: this_path, 
                     self.meta_t: this_meta, 
                     self.dest_t: this_dest,
                     self.keep_t: FLAGS.keep_prob}
        _, train_loss = self.sess.run([self.train_op, self.loss_t], feed_dict=feed_dict)
        step = self.latest_step

        # Time consumption summaries
        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % log_freq == 0:

          # Validation loss
          current_loss = self.sess.run(self.loss_t, feed_dict=feed_dict_val)

          # Print status to stdout.
          print('\rStep %d: trn_loss=%.2f, val_loss=%.2f (%.1f sec/batch)'
                % (step, train_loss, current_loss, duration), flush=True, end='\r')

          # Save checkpoint if current loss is the best
          if best_loss is None or (current_loss < best_loss):
            print('')
            best_loss, best_step = current_loss, step
            self._save_ckpt(step)

          # Early stoppping          
          delta_steps = step - best_step
          stop_now = delta_steps >= early_stopping_rounds * log_freq
          if stop_now:
            print('')
            log.warning('Stopping. Best step: {} with {:.4f}.'
                        .format(best_step, best_loss))
            coord.request_stop()

    except KeyboardInterrupt:
        print('\n>> Training stopped by user! Restore the latest ckpt.')
    
    # restore params from the latest ckpt
    self.init_or_restore_all_variables(restart=False)


  def predict(self, path, meta):
    """return destination predicted by this model"""
    assert self.latest_checkpoint is not None

    feed_dict = {self.path_t: path, self.meta_t: meta, self.keep_t: 1}
    pred_v = self.sess.run(self.pred_t, feed_dict=feed_dict)
    
    return pred_v


  def eval_dist(self, path, meta, dest):
    """return destination predicted by this model"""
    assert self.latest_checkpoint is not None

    feed_dict = {self.path_t: path, self.meta_t: meta, self.dest_t: dest, self.keep_t: 1}
    dist_v = self.sess.run(self.distances, feed_dict=feed_dict)
    
    return dist_v



  def eval_mean_distance(self, path, meta, dest):
    """return evaluation mectrics calculated by this model for the given dataset
    """
    assert self.latest_checkpoint is not None

    feed_dict = {self.path_t: path, 
                 self.meta_t: meta, 
                 self.dest_t: dest,
                 self.keep_t: 1}
    return self.sess.run(self.average_loss_t, feed_dict=feed_dict)


  def close_session(self):
    self.sess.close()