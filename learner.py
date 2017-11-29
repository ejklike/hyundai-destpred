import os
import time

import numpy as np
import tensorflow as tf

from log import log
from utils import maybe_exist
from tf_utils import (BatchGenerator, 
                      NeighborWeightCalculator, 
                      compute_km_distances,
                      ModifiedMeanShift)
from graph import (extract_feature)

# FLAGS to be applied across related modules
FLAGS = tf.flags.FLAGS


class Model(object):

  def __init__(self, model_dir):
    # Get model_dir
    maybe_exist(model_dir)
    self.model_dir = model_dir
    log.info('model_dir: %s', model_dir)

  def prepare_prediction(self, dest_trn):
    """
    # We want weighted loss to ignore infrequent destinations 
    # and to predict frequent destinations well.
    # The reference points will be training destinations.
    """
    # Define loss_weight_calculator
    self.loss_weight_gen = NeighborWeightCalculator(radius=FLAGS.radius, 
                                                    reference_points=dest_trn)
    # Find centroids of destinations
    if FLAGS.cband > 0:
      self.clustering = ModifiedMeanShift(bandwidth=FLAGS.cband, min_freq=5).fit(dest_trn)
      self.centroids = self.clustering.cluster_centers_

      cluster_labels = self.clustering.predict(dest_trn) # starting from 0
      self.cluster_counts = np.bincount(cluster_labels)
      print(self.cluster_counts)

      print('cband=%f, #cluster=%d' %(FLAGS.cband, len(self.centroids)))

    else:
      self.centroids = None


  def build_graph(self):

    # Parameters
    learning_rate = FLAGS.learning_rate

    # Reset default_graph
    tf.reset_default_graph()

    # with tf.Graph().as_default():
    # Define global_step
    self.global_step = tf.train.get_or_create_global_step()

    # Placeholders
    path_shape = dict(
        dnn=[None, 2 * 2 * FLAGS.k], # fixed size
        rnn=[None, FLAGS.max_length, 2] # variable length
    )[FLAGS.model_type]
    self.path_t = tf.placeholder(dtype=tf.float32, 
                                 shape=path_shape, # defined by model_type
                                 name='path_placeholder')
    self.meta_t = tf.placeholder(dtype=tf.int32, 
                                 shape=[None, 4], 
                                 name='meta_placeholder')
    self.dest_t = tf.placeholder(dtype=tf.float32, # for prediction
                                 shape=[None, 2], 
                                 name='dest_placeholder')
    self.label_t = tf.placeholder(dtype=tf.float32, # for classification
                                 shape=[None, ], 
                                 name='label_placeholder')
    self.loss_weight_t = tf.placeholder(dtype=tf.float32,
                                        shape=[None, ],
                                        name='loss_weight_placeholder')

    # Building graph from placeholders to extract hidden feature
    feature_t = extract_feature(self.path_t, self.meta_t)

    # Stack the last layer 
    # to predict or clssify the final destination

    # Classification
    if FLAGS.classification is True:
      self.logit_t = tf.layers.dense(feature_t, len(self.centroids), 
                                     activation=None, name='logit')
      self.label_t = tf.to_int32(self.label_t)

      label_weight = tf.constant(1 / self.cluster_counts, 
                                 dtype=tf.float32, name='label_weight')
      batch_weight = tf.nn.embedding_lookup(label_weight, self.label_t, name='loss_weight')

      xe_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_t,
                                                                 logits=self.logit_t,
                                                                 name='xe_losses')
      weighted_xe_losses = tf.multiply(xe_losses, batch_weight, name='weighted_xe_losses')
      self.xe_loss_t = tf.reduce_mean(xe_losses, name='xentropy_mean')
      tf.add_to_collection(tf.GraphKeys.LOSSES, self.xe_loss_t)

    # Prediction
    else:
      # centroids-weighted prediction
      if self.centroids is not None:
          centroids_t = tf.constant(self.centroids,
                                    dtype=tf.float32, 
                                    name='centroids_constant')
          probs_t = tf.layers.dense(feature_t, len(self.centroids), 
                                    activation=tf.nn.softmax, 
                                    name='cluster_probs')
          self.pred_t = tf.nn.xw_plus_b(probs_t, centroids_t, [0., 0.], name='final')

      # Direct prediction
      else:
        self.pred_t = tf.layers.dense(feature_t, 2, activation=None, name='fianl')

      # Define prediction loss
      squared_distances = compute_km_distances(self.dest_t, self.pred_t)
      weights = self.loss_weight_t    # for weighted average
      self.weighted_loss_t = tf.losses.compute_weighted_loss(squared_distances, weights, 
                                                            scope='weighted_loss', 
                                                            loss_collection=tf.GraphKeys.LOSSES)
      weights = 1.0
      self.average_loss_t = tf.losses.compute_weighted_loss(squared_distances, weights, 
                                                            scope='average_loss', 
                                                            loss_collection=None)

    # Gather all losses
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    # losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss_t = tf.add_n(losses, name='training_loss')

    # Training Op.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = optimizer.minimize(loss=self.loss_t, 
                                       global_step=tf.train.get_global_step())

    # Initializing Op.
    # self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    self.init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    self.saver = tf.train.Saver(max_to_keep=1)

    # Create a session for running Ops on the Graph.
    sess_config = tf.ConfigProto()
    if FLAGS.gpu_mem_frac < 1:
      sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem_frac
    else:
      sess_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
    self.sess = tf.Session(config=sess_config)


  def init_or_restore_all_variables(self, restart=False):
    # Remove prev model or not
    if restart is True and tf.gfile.Exists(self.model_dir):
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
    batch_size = FLAGS.batch_size
    early_stopping_rounds = FLAGS.early_stopping_rounds
    log_freq = FLAGS.log_freq

    # Split train set into train/validation sets
    num_trn = int(len(path) * (1 - FLAGS.validation_size))
    path_trn, path_val = path[:num_trn], path[num_trn:]
    meta_trn, meta_val = meta[:num_trn], meta[num_trn:]
    dest_trn, dest_val = dest[:num_trn], dest[num_trn:]

    # Instantiate batch generator
    trn_batch_generator = BatchGenerator([path_trn, meta_trn, dest_trn], batch_size)

    # Define feed_dict for validation and early stopping
    loss_weights_val = self.loss_weight_gen.get_neighbor_weight(dest_val)
    if FLAGS.classification:
      label_val = self.clustering.predict(dest_val)#.astype(np.int32)
      feed_dict_val = {self.path_t: path_val, 
                       self.meta_t: meta_val, 
                       self.label_t: label_val}
    else:
      feed_dict_val = {self.path_t: path_val, 
                       self.meta_t: meta_val, 
                       self.dest_t: dest_val,
                       self.loss_weight_t: loss_weights_val}

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
        if FLAGS.classification:
          this_label = self.clustering.predict(this_dest)
          # print(this_label, this_label.dtype, len(this_label), np.max(this_label))
          feed_dict = {self.path_t: this_path, 
                      self.meta_t: this_meta, 
                      self.label_t: this_label}
          _, train_loss = self.sess.run([self.train_op, self.loss_t], feed_dict=feed_dict)
        else:
          loss_weights = self.loss_weight_gen.get_neighbor_weight(this_dest)
          feed_dict = {self.path_t: this_path, 
                      self.meta_t: this_meta, 
                      self.dest_t: this_dest,
                      self.loss_weight_t: loss_weights}
          _, train_loss = self.sess.run([self.train_op, self.loss_t], feed_dict=feed_dict)
        step = self.latest_step

        # Time consumption summaries
        duration = time.time() - start_time
        examples_per_sec = batch_size / duration

        # Write the summaries and print an overview fairly often.
        if step % log_freq == 0:

          # Validation loss
          current_loss = self.sess.run(self.loss_t, feed_dict=feed_dict_val)
          
          # Print status to stdout.
          print('\rStep %d: trn_loss=%.2f, val_loss=%.2f (%.1f sec/batch; %.1f examples/sec)'
                % (step, train_loss, current_loss, duration, examples_per_sec),
                flush=True, end='\r')

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

          # # Update the events file.
          # summary_str = self.sess.run(summary_op)
          # summary_writer.add_summary(summary_str, step)

    except KeyboardInterrupt:
        print('\n>> Training stopped by user!')


  def predict(self, path, meta, dest):
    """return destination predicted by this model"""
    assert self.latest_checkpoint is not None

    if FLAGS.classification:
      label = self.clustering.predict(dest)
      feed_dict = {self.path_t: path,
                   self.meta_t: meta, 
                   self.label_t: label}
      logit_v = self.sess.run(self.logit_t, feed_dict=feed_dict)

      argmax_cluster = np.argmax(logit_v, axis=1)
      pred_v = self.centroids[argmax_cluster]
    
    else:

      feed_dict = {self.path_t: path, self.meta_t: meta}
      pred_v = self.sess.run(self.pred_t, feed_dict=feed_dict)
    
    return pred_v


  def eval_metrics(self, path, meta, dest):
    """return evaluation mectrics calculated by this model for the given dataset
    """
    assert self.latest_checkpoint is not None
    if FLAGS.classification:
      pred = self.predict(path, meta, dest)

      # 
      anomaly_mask = (self.clustering.predict(dest) == 0).astype(np.float32).reshape(-1, 1)
      pred = np.multiply(1 - anomaly_mask, pred) + np.multiply(anomaly_mask, dest)

      from utils import dist
      return dist(pred, dest, to_km=True), dist(pred, dest, to_km=True)
    else:
      loss_weights = self.loss_weight_gen.get_neighbor_weight(dest)
      feed_dict = {self.path_t: path, 
                  self.meta_t: meta, 
                  self.dest_t: dest,
                  self.loss_weight_t: loss_weights}
      return self.sess.run([self.average_loss_t, self.weighted_loss_t], 
                          feed_dict=feed_dict)


  def close_session(self):
    self.sess.close()