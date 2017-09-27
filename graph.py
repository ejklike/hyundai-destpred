import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape):
    initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _fully_conn_layer(x, out_dim, activation_fn=tf.nn.relu, name=None):
    name = 'fully_connected' if name is None else name
    dim = x.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        weights = _variable_on_cpu('weights', [dim, out_dim])
        biases = _variable_on_cpu('biases', [out_dim])
        if activation_fn:
            x = activation_fn(tf.matmul(x, weights) + biases, name=scope.name)
        else:
            x = tf.add(tf.matmul(x, weights), biases, name=scope.name)
    _activation_summary(x)
    return x


def rnn_last_output(rnn_input, cell_type='rnn', n_unit=16, bi_direction=False):
    cell_dict = dict(
        rnn=rnn.BasicRNNCell,
        lstm=rnn.BasicLSTMCell
    )
    rnn_cell = cell_dict[cell_type](n_unit)
    
    if bi_direction:
        outputs, _ = rnn.static_rnn(
            rnn_cell, rnn_input, dtype=tf.float32)
    else:
        outputs, _, _ = rnn.static_bidirectional_rnn(
            rnn_cell, rnn_cell, rnn_input, dtype=tf.float32)
    return outputs[-1]


def inputs(paths, metas, dests, test=False):
    """
    args:
        paths: list of path nparrays
        metas: list of meta lists
        dests: list of dest nparrays
    """
    if FLAGS.is_rnn:
        max_length = max(p.shape[0] for p in paths)
        def resize_by_padding(path, target_length):
            """add zero padding prior to the given path (np array)
            """
            path_length = path.shape[0]
            pad_width = ((target_length - path_length, 0), (0, 0))
            return np.lib.pad(path, pad_width,
                              'constant', constant_values=0)
        
        paths = [resize_by_padding(p, max_length) for p in paths]
        paths = np.stack(paths, axis=0)

    else:
        k = 10
        def resize_to_2k(path, k):
            """remove middle portion of the given path (np array)
            """
            return np.concatenate([path[:k], path[-k:]], axis=0)
        
        paths = [resize_to_2k(path, FLAGS.k) for p in paths]
        paths = np.stack(paths, axis=0)
    
    dests = np.array(dests)

    num_epochs = 1 if test else FLAGS.num_epochs
    allow_smaller_final_batch = True if test else False

    with tf.device('/cpu:0'):
        paths = tf.constant(paths, dtype=tf.float32)
        metas = tf.constant(metas, dtype=tf.int32)
        dests = tf.constant(dests, dtype=tf.float32)

    if test is False:
        path, meta, dest = tf.train.slice_input_producer(
            [paths, metas, dests], num_epochs=num_epochs)
        
        # the maximum number of elements in the queue
        capacity = 20 * FLAGS.batch_size
        paths, metas, dests = tf.train.batch(
            [path, meta, dest], 
            batch_size=FLAGS.batch_size, 
            num_threads=FLAGS.num_threads,
            capacity=capacity,
            allow_smaller_final_batch=allow_smaller_final_batch)
    
    return paths, metas, dests


def inference(paths, metas):
    n_hidden = 16
    embedding_dim = 5

    # PATH embedding
    if FLAGS.is_rnn:
        rnn_inputs = tf.unstack(paths, axis=1)
        path_embedding = rnn_last_output(
            rnn_inputs, 
            cell_type='lstm', 
            n_unit=16, 
            bi_direction=False)
    else:
        nn_inputs = tf.reshape(paths, shape=[FLAGS.batch_size, -1])
        path_embedding = _fully_conn_layer(nn_inputs, n_hidden, name='path_embedding')

    # META embedding
    holiday, weekno, hour, weekday = tf.split(metas, metas.shape[1], axis=1)
    
    holi_W = _variable_on_cpu('holiday_embedding', [2, 1])
    holiday_embedding = tf.nn.embedding_lookup(holi_W, holiday)
    holiday_embedding = tf.squeeze(holiday_embedding, axis=1)

    weekno_W = _variable_on_cpu('weekno_embedding', [53, 10])
    weekno_embedding = tf.nn.embedding_lookup(weekno_W, weekno)
    weekno_embedding = tf.squeeze(weekno_embedding, axis=1)

    hour_W = _variable_on_cpu('hour_embedding', [24, 4])
    hour_embedding = tf.nn.embedding_lookup(hour_W, hour)
    hour_embedding = tf.squeeze(hour_embedding, axis=1)

    weekday_W = _variable_on_cpu('weekday_embedding', [7, 2])
    weekday_embedding = tf.nn.embedding_lookup(weekday_W, weekday)
    weekday_embedding = tf.squeeze(weekday_embedding, axis=1)

    # CONCAT
    concatenation = tf.concat([path_embedding, 
                               holiday_embedding, 
                               weekno_embedding, 
                               hour_embedding,
                               weekday_embedding], axis=1, name='concat')
    
    # FINAL
    h = _fully_conn_layer(concatenation, n_hidden, name='final_dense')
    return _fully_conn_layer(h, 2, activation_fn=None, name='last_linear')


def loss(preds, dests):
  return tf.losses.mean_squared_error(dests, preds, scope='mse_loss')

def train(total_loss, global_step):
  opt = tf.train.RMSPropOptimizer(FLAGS.lr)
  train_op = opt.minimize(total_loss, global_step=global_step)
  return train_op