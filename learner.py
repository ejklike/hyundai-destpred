import os
import time

import numpy as np
import tensorflow as tf

from log import log
from utils import maybe_exist, dist
from tf_utils import (BatchGenerator, 
                      compute_km_distances)
from clustering import ModifiedMeanShift
from sklearn.metrics.pairwise import pairwise_distances
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
    # Find centroids of destinations
    if FLAGS.cband > 0:
      self.clustering = ModifiedMeanShift(radius=1.2, bandwidth=FLAGS.cband, 
                                          major_min_freq=50, minor_min_freq=5, 
                                          cluster_all=True).fit(dest_trn)
      # self.clustering = MeanShiftWrapper(bandwidth=0.3, min_freq=30).fit(dest_trn)
      self.cluster_centers_ = self.clustering.cluster_centers_
      self.cluster_counts_ = self.clustering.cluster_counts_
      self.noise_count_ = np.sum(self.clustering.labels_==-1, dtype=np.int32)

      print('cband=%f, #cluster=%d' %(FLAGS.cband, len(self.cluster_centers_)), 
            '#noise=', self.noise_count_)
      print('cluster distribution', self.cluster_counts_)

    else:
      self.cluster_centers_ = None


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
    # self.dest_t = tf.placeholder(dtype=tf.float32, # for prediction
    #                              shape=[None, 2], 
    #                              name='dest_placeholder')
    self.label_t = tf.placeholder(dtype=tf.float32, # for classification
                                 shape=[None, ], 
                                 name='label_placeholder')

    # Building graph from placeholders to extract hidden feature
    feature_t = extract_feature(self.path_t, self.meta_t)

    # Stack the last layer 
    # to predict or clssify the final destination

    # CLASSIFICATION
    # self.logit_t = tf.layers.dense(feature_t, 1 + self.clustering.n_cluster_, 
    self.logit_t = tf.layers.dense(feature_t, self.clustering.n_cluster_, 
                                    activation=None, name='logit')
    self.label_t = tf.to_int32(self.label_t)

    # # losses: [batch_size, ]
    xe_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_t,
                                                               logits=self.logit_t,
                                                               name='xe_losses')


    # original
    batch_weight = 1.0      
    
    # custom 1: cost by class counts
    # label_weight = 1 / tf.constant([self.noise_count_] + list(self.cluster_counts_), 
    #                            dtype=tf.float32, name='label_weight')
    # batch_weight = tf.nn.embedding_lookup(label_weight, self.label_t, name='loss_weight')
    
    # # custom 2: cost proportion to the distance between classes
    # self.pair_dist_mat = pairwise_distances(self.cluster_centers_, 
    #                                         metric='wminkowski', p=2, w=[88.8, 111.0]) + 1
    # dist_mat_t = tf.constant(self.pair_dist_mat, name='pairwise_distance', dtype=tf.float32)
    # batch_label_weight = tf.nn.embedding_lookup(dist_mat_t, self.label_t, name='batch_label_weight')
    # one_hot_mask = tf.one_hot(tf.argmax(self.logit_t, axis=1), depth=self.clustering.n_cluster_)
    # batch_label_weight = tf.reduce_sum(tf.multiply(batch_label_weight, one_hot_mask), axis=1)
    
    # for original and custom 1
    weighted_xe_losses = tf.multiply(xe_losses, batch_weight, name='weighted_xe_losses')

    # for custom 2
    # weighted_xe_losses = tf.multiply(xe_losses, batch_label_weight, name='weighted_xe_losses')
    # weighted_xe_losses = xe_losses + 0.01 * batch_label_weight

    # Final Cost
    self.xe_loss_t = tf.reduce_mean(weighted_xe_losses, name='xentropy_mean') # weighted_xe_losses
    tf.add_to_collection(tf.GraphKeys.LOSSES, self.xe_loss_t)

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
    # self.init_op = tf.global_variables_initializer()

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
    
    # Initialize OR restore variables
    if self.latest_checkpoint is None:
      self._init_all_trainable_variables()
    else:
      self._restore_all_trainable_variables(self.latest_checkpoint)


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
    # label_val = self.clustering.predict(dest_val) + 1
    label_val = self.clustering.predict(dest_val)
    feed_dict_val = {self.path_t: path_val, 
                      self.meta_t: meta_val, 
                      self.label_t: label_val}

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
        # this_label = self.clustering.predict(this_dest) + 1
        this_label = self.clustering.predict(this_dest)
        # print(this_label, this_label.dtype, len(this_label), np.max(this_label))
        feed_dict = {self.path_t: this_path, 
                    self.meta_t: this_meta, 
                    self.label_t: this_label}
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

    true_cluster, pred_cluster = self.get_true_pred(path, meta, dest)
    pred_v = self.cluster_centers_[pred_cluster]
 
    return pred_v


  def get_true_pred(self, path, meta, dest):
    """return true and pred destination labels predicted by this model"""
    assert self.latest_checkpoint is not None

    # label = self.clustering.predict(dest) + 1
    label = self.clustering.predict(dest)
    feed_dict = {self.path_t: path,
                  self.meta_t: meta, 
                  self.label_t: label}
    logit_v = self.sess.run(self.logit_t, feed_dict=feed_dict)

    cluster_label = self.clustering.predict(dest)
    cluster_label_hat = np.argmax(logit_v, axis=1)

    return cluster_label, cluster_label_hat


  def eval_metrics(self, path, meta, dest):
    """return evaluation mectrics calculated by this model for the given dataset
    """
    assert self.latest_checkpoint is not None

    true, pred = self.get_true_pred(path, meta, dest)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix

    accuracy = accuracy_score(true, pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(true, pred, average='macro')

    from utils import dist

    # filter_idxs = np.logical_and(true != -1, pred != -1)
    # filter_labels = pred[filter_idxs]
    # mean_dist = dist(dest[filter_idxs], self.cluster_centers_[filter_labels], to_km=True)
    mean_dist = dist(dest, self.cluster_centers_[pred], to_km=True)

    # def convert_to_binary_noise(x):
    #   internal_x = x.copy()
    #   internal_x[internal_x!=-1] = 0 # remove all cluster informations
    #   return -internal_x # noise becomes 1

    # true_noise, pred_noise = convert_to_binary_noise(true), convert_to_binary_noise(pred)
    # _, _, fscore_noise, _ = precision_recall_fscore_support(true_noise, pred_noise, average=None)
    # fscore_noise = fscore_noise[1]



    print('ACC, PREC, REC, F1: {:.2f}, {:.2f}, {:.2f}, {:.2f}'#, F1_NOISE: {:.2f}'
          .format(accuracy, precision, recall, fscore))#, fscore_noise))
    print('MEAN DISTANCE ERROR: {:.3f}km'.format(mean_dist))
    # print('NOISE PREC, REC, F1: {:.2f}, {:.2f}, {:.2f}'.format(pre_noise, rec_noise, fsc_noise))
    # print('CONFUSION_MATRIX: ')
    # print(confusion_matrix(true, pred))

    from sklearn.metrics import classification_report
    target_names = ['C' + str(i) for i in range(self.clustering.n_cluster_)]
    print(classification_report(true, pred, target_names=target_names))

    return [accuracy, precision, recall, fscore, mean_dist]#, fscore_noise]


  # def eval_metrics_backup(self, path, meta, dest):
  #   """return evaluation mectrics calculated by this model for the given dataset
  #   """
  #   assert self.latest_checkpoint is not None
  #   if FLAGS.classification:
  #     pred = self.predict(path, meta, dest) + 1

  #     # 
  #     anomaly_mask = (self.clustering.predict(dest) == -1).astype(np.float32).reshape(-1, 1)
  #     pred = np.multiply(1 - anomaly_mask, pred) + np.multiply(anomaly_mask, dest)

  #     from utils import dist
  #     return dist(pred, dest, to_km=True), dist(pred, dest, to_km=True)
  #   else:
  #     loss_weights = self.loss_weight_gen.get_neighbor_weight(dest)
  #     feed_dict = {self.path_t: path, 
  #                 self.meta_t: meta, 
  #                 self.dest_t: dest}
  #     return self.sess.run([self.average_loss_t, self.weighted_loss_t], 
  #                         feed_dict=feed_dict)


  def close_session(self):
    self.sess.close()