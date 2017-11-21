from math import ceil
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from log import log
from custom_loss import neg_log_likelihood_loss
from seq2seq_model import infer_params, loss_and_train_op, predict

DTYPE = tf.float32



class BatchGenerator(object):

  def __init__(self, data_list, batch_size, epoch=None):
    self.pointer = 0
    self.counter = 0
    self.batch_size = batch_size
    self.epoch = epoch
    self.data_size = data_list[0].shape[0]
    self.data_list = [np.concatenate([data, data], axis=0) for data in data_list]

  def next_batch(self):
    if (self.epoch is not None) and (self.counter > self.data_size * self.epoch):
      raise tf.errors.OutOfRangeError('OutOfRange ERROR!')

    next_pointer = self.pointer + self.batch_size

    batch_list = [data[self.pointer:next_pointer] for data in self.data_list]
    self.counter += (next_pointer - self.pointer)
    self.pointer = (next_pointer) % self.data_size

    return batch_list


class Model(object):

  def __init__(self, params, model_dir):
    self.params = params
    self.model_dir = model_dir
    print(model_dir)

    self.global_step = tf.train.get_or_create_global_step()

    # Create a session for running Ops on the Graph.
    sess_config = tf.ConfigProto()
    if params['gpu_mem_frac'] < 1:
      sess_config.gpu_options.per_process_gpu_memory_fraction = params['gpu_mem_frac']
    else:
      sess_config.gpu_options.allow_growth = params['gpu_allow_growth']
    self.sess = tf.Session(config=sess_config)

    # Placeholders
    self.input_t = tf.placeholder(dtype=tf.float32, 
                                  shape=[None, None, 3], 
                                  name='input_placeholder')
    self.output_t = tf.placeholder(dtype=tf.float32, 
                                   shape=[None, None, 3], 
                                   name='target_placeholder')
    self.input_t1 = tf.placeholder(dtype=tf.float32, 
                                   shape=[None, 1, 3], 
                                   name='input_placeholder_tst')

    # CELL definition
    self.cell = rnn.BasicLSTMCell(self.params['rnn_size'], state_is_tuple=True)
    # cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)
    self.state_in = self.cell.zero_state(batch_size=params['batch_size'], dtype=DTYPE)
    # state_in = tf.identity(zero_state, name='state_in')
    self.state_in1 = self.cell.zero_state(batch_size=1, dtype=DTYPE)
    # state_in_tst = tf.identity(zero_state_tst, name='state_in_tst')

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    with tf.variable_scope('infer') as scope:
      *self.params_out, self.state_out = infer_params(self.input_t, 
                                                      self.cell, 
                                                      self.state_in, 
                                                      self.params)
      self.loss_t, self.train_op = loss_and_train_op(self.output_t, 
                                                     *self.params_out, 
                                                     learning_rate=self.params['learning_rate'])
      scope.reuse_variables()
      *self.params_out1, self.state_out1 = infer_params(self.input_t1, 
                                                        self.cell, 
                                                        self.state_in1, 
                                                        self.params)
      self.mu, self.cov, self.eop = predict(*self.params_out1, 
                                            self.params)

    # Create and run the Op to initialize the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    self.sess.run(init_op)


  def _generate_feed_dict(self, batch_generator):
    """Generate feed_dict from known batch_list."""
    this_input, this_target = batch_generator.next_batch()
    return {self.input_t: this_input, self.output_t: this_target}


  def _save_ckpt(self, step):
    """Saves the latest checkpoint."""
    log.info("...Saving checkpoints for %d.", step)
    save_path = os.path.join(self.model_dir, 'model.ckpt')
    self.saver.save(self.sess, save_path, global_step=step)


  def _validation_loss(self, batch_generator):
    """Calculate validation loss."""
    iteration_size = ceil(batch_generator.data_size / batch_generator.batch_size)
    feed_dict_list = [self._generate_feed_dict(batch_generator) for _ in range(iteration_size)]
    losses = [self.sess.run(self.loss_t, feed_dict=feed_dict) for feed_dict in feed_dict_list]
    return np.sum(losses) / iteration_size


  def train(self, input_data, target_data, restart=False):
    """train the model"""

    # Create a saver for writing training checkpoints.
    self.saver = tf.train.Saver(max_to_keep=1)

    # Remove prev model or not
    if tf.gfile.Exists(self.model_dir):
      if restart is True:
        log.warning('Delete prev model_dir: %s', self.model_dir)
        tf.gfile.DeleteRecursively(self.model_dir)
      else:
        ckpt_path = tf.train.latest_checkpoint(self.model_dir)
        log.info('Ckpt path: %s', ckpt_path)
        if ckpt_path is not None:
          log.info('Restore from the ckpt path.')
          self.saver.restore(self.sess, ckpt_path)

    # # Instantiate a SummaryWriter to output summaries and the Graph.
    # summary_writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

    # split train set into train/validation sets
    num_trn = int(len(input_data) * (1 - self.params['validation_size']))
    input_trn, input_val = input_data[:num_trn], input_data[num_trn:]
    target_trn, target_val = target_data[:num_trn], target_data[num_trn:]
    print('input data shape of (trn, val): ', input_trn.shape, input_val.shape)

    # Training batch
    trn_batch_gen = BatchGenerator([input_trn, target_trn], self.params['batch_size'])
    val_batch_gen = BatchGenerator([input_val, target_val], self.params['batch_size'])

    # Start input enqueue threads.
    coord = tf.train.Coordinator()

    # # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.summary.merge_all()

    # And then after everything is built, start the training loop.
    try:
      best_loss, best_loss_step = None, 0
      while not coord.should_stop():
        
        start_time = time.time()

        # Run one step of the model.
        _, train_loss, step = self.sess.run([self.train_op, self.loss_t, self.global_step], 
                                            feed_dict=self._generate_feed_dict(trn_batch_gen))
        
        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % self.params['log_freq'] == 0:

          # Do Validation and Print status to stdout.
          current_loss = self._validation_loss(val_batch_gen)
          examples_per_sec = self.params['batch_size'] / duration
          print('\rStep %d: trn_loss=%.2f, val_loss=%.2f (%.2f sec/batch; %.1f examples/sec)'
                % (step, train_loss, current_loss, duration, examples_per_sec),
                flush=True, end='\r')

          # Save checkpoint
          if best_loss is None or (current_loss < best_loss):
            print('')
            best_loss, best_loss_step = current_loss, step
            self._save_ckpt(step)

          # Early stoppping          
          delta_steps = step - best_loss_step
          stop_now = delta_steps >= self.params['early_stopping_rounds'] * self.params['log_freq']
          if stop_now:
            print('')
            log.warning('Stopping. Best step: {} with {:.4f}.'
                        .format(best_loss_step, best_loss))
            coord.request_stop()

          # # Update the events file.
          # summary_str = self.sess.run(summary_op)
          # summary_writer.add_summary(summary_str, step)

    except KeyboardInterrupt:
        print('\n>> Training stopped by user!')


  def get_test_input_state(self):
    return self.sess.run(self.cell.zero_state(batch_size=1, dtype=DTYPE))


  def predict_next(self, prev_input, prev_state):
    """
    # WARNING: THIS CODE ONLY WORKS WITH BATCH SIZE 1 !!!
    # batch_size MUST BE "1" for valid prediction!!!!
    """
    feed_dict = {self.state_in1: prev_state, self.input_t1: prev_input}
    *params_out1_v, mu_v, cov_v, eop_v, next_state = self.sess.run(
        [*self.params_out1, self.mu, self.cov, self.eop, self.state_out1], 
        feed_dict=feed_dict)
    
    for k, v in zip(self.params_out1, params_out1_v):
      print(k, ':', v)
    
    def sample_gaussian_2d(mu, cov):
      return np.random.multivariate_normal(mu, cov, 1)[0]

    xy = sample_gaussian_2d(mu_v, cov_v)
    # xy = mu_v
    eop_v = int(eop_v)
    next_input = np.array([xy[0], xy[1], eop_v], dtype=np.float32)
    return next_input, next_state

  def close_session(self):
    self.sess.close()