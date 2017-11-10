from datetime import datetime
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import meta_graph
from tensorflow.python.training import training_util
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training.summary_io import SummaryWriterCache

from log import log


class EarlyStoppingHook(tf.train.SessionRunHook):

    def __init__(self, 
                 log_freq=100, 
                 early_stopping_rounds=10, 
                 checkpoint_basename="model.ckpt",
                 checkpoint_dir=None):
        """ control earlystopping flow & save checkpoint """
        log.info("Create EarlyStoppingHook to save the best checkpoint.")

        # control interval
        self.log_freq = log_freq
        self.early_stopping_steps = early_stopping_rounds * log_freq

        # early stopping
        self._best_value = None
        self._best_value_step = None
        
        # save checkpoint
        self._saver = None
        self._scaffold = None
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)

        

    def _get_saver(self):
        # Get saver from the SAVERS collection if present.
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor.".
                format(collection_key))
        self._saver = savers[0]
        return savers[0]

    def begin(self):
        # You can add ops to the graph here.
        log.info('>>> Starting the session.')
        self._step = -1
        self._start_time = datetime.now()
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")

    def before_run(self, run_context):
        self._step += 1
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        if self._step % self.log_freq == 0:
            training_util.write_graph(
                ops.get_default_graph().as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt")
            saver_def = self._get_saver().saver_def if self._get_saver() else None
            graph = ops.get_default_graph()
            meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=graph.as_graph_def(add_shapes=True),
                saver_def=saver_def)
            self._summary_writer.add_graph(graph)
            self._summary_writer.add_meta_graph(meta_graph_def)

        # global_step = tf.train.get_global_step()
        global_step = self._global_step_tensor
        train_loss = tf.get_collection(tf.GraphKeys.LOSSES)[0]

        graph = run_context.session.graph
        valid_loss = graph.get_tensor_by_name('loss_val/Mean:0')
        return tf.train.SessionRunArgs([global_step, train_loss, valid_loss])

    def _save_ckpt(self, session):
        """Saves the latest checkpoint."""
        log.info("...Saving checkpoints for %d.", self._best_value_step)
        
        self._get_saver().save(session, self._save_path,
                               global_step=self._best_value_step)
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
            self._best_value_step)

    def after_run(self, run_context, run_values):
        if self._step % self.log_freq == 0:
            global_step, train_loss, valid_loss = run_values.results
            elapsed_time_min = (datetime.now() - self._start_time).total_seconds()
            print('\rGlobal step: %d, Train_loss: %.4f, Valid_loss: %.4f (%d sec for %d steps)'
                  %(global_step, train_loss, valid_loss, elapsed_time_min, self.log_freq), 
                  end='\r', flush=True)

            current_value, current_step = valid_loss, global_step
            if self._best_value is None or (current_value < self._best_value):
                print('')
                self._best_value = current_value
                self._best_value_step = current_step
                self._save_ckpt(run_context.session)
            delta_steps = current_step - self._best_value_step
            stop_now = (delta_steps >= self.early_stopping_steps)
            if stop_now:
                print('')
                log.warning('Stopping. Best step: {} with {:.4f}.'
                            .format(self._best_value_step, self._best_value))
                run_context.request_stop()
            self._start_time = datetime.now()

    def end(self, session):
        print('')
        log.info('>>> Done with the session.')


def ValidHook():
    self._global_step_tensor = training_util._get_or_create_global_step_read()