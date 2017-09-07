import tensorflow as tf
from tensorflow.contrib import layers, rnn

from sklearn.cluster import MeanShift

def feed_forward_hidden(h, n_hidden_list=None):
    if n_hidden_list is not None:
        for n_hidden in n_hidden_list:
            h = layers.fully_connected(h, n_hidden)
    return h

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
#     h = outputs[-1]
#     return layers.linear(h, 2)
    return outputs[-1]

class Model(object):
    def __init__(self, is_rnn=False, model_param=None):
        assert model_param is not None
        
        seq_len = model_param['seq_len']
        n_hidden_list = model_param['n_hidden_list']
        if is_rnn:
            cell_type = model_param['cell_type']
            n_unit = model_param['n_unit']
            bi_direction = model_param['bi_direction']
        
        tf.reset_default_graph()

        self.x_input = tf.placeholder(tf.float32, [None, seq_len*2])
        self.y_true = tf.placeholder(tf.float32, [None, 2])

        if is_rnn:
            x_reshaped = tf.reshape(self.x_input, shape=[-1, seq_len, 2])
            rnn_input = tf.unstack(x_reshaped, axis=1)
            nn_input = rnn_last_output(
                rnn_input, cell_type=cell_type, n_unit=n_unit, 
                bi_direction=bi_direction)
        else:
            nn_input = self.x_input
        
        self.last_hidden = feed_forward_hidden(nn_input, 
                                          n_hidden_list=n_hidden_list)

    def open_session(self, use_gpu=False):
        gpu_cpu_conf = tf.ConfigProto(
            device_count = {'GPU': 1 if use_gpu else 0})
        self.sess = tf.Session(config=gpu_cpu_conf)

    def fit(self, x_trn, x_tst, y_trn, y_tst, 
            is_regression=True, learning_param=None):
        assert learning_param is not None

        learning_rate = learning_param['learning_rate']
        max_iter = learning_param['max_iter']

        if is_regression:
            self.y_pred = layers.linear(self.last_hidden, 2)
        else:
            mean_shift = MeanShift().fit(y_trn)
            cluster_centers = mean_shift.cluster_centers_
            n_cluster = len(cluster_centers)
            print('#cluster of destination = ', n_cluster)
            
            c_logits = layers.linear(self.last_hidden, n_cluster)
            c_probs = layers.softmax(c_logits)
            c_centers = tf.constant(cluster_centers, dtype=tf.float32)
            self.y_pred = tf.nn.xw_plus_b(c_probs, c_centers, [0., 0.])

        self.mse_loss = tf.losses.mean_squared_error(self.y_true, self.y_pred)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(self.mse_loss)
        
        tf.global_variables_initializer().run(session=self.sess)

        def _trn_feed_dict(is_training):
            if is_training:
                return {self.x_input: x_trn, self.y_true: y_trn}
            else:
                return {self.x_input: x_tst, self.y_true: y_tst}

        # Train
        self.trn_loss_list, self.tst_loss_list = [], []
        for _ in range(max_iter):
            _, trn_loss = self.sess.run([train_step, self.mse_loss], 
                                   feed_dict=_trn_feed_dict(True))
            tst_loss = self.sess.run(self.mse_loss, 
                                feed_dict=_trn_feed_dict(False))
            self.trn_loss_list.append(trn_loss)
            self.tst_loss_list.append(tst_loss)


        print('trn mse--->', self.trn_loss_list[-1])
        print('tst mse--->', self.tst_loss_list[-1])
    
    def predict(self, x):
        return self.sess.run(self.y_pred, feed_dict={self.x_input: x})

    def close_session(self):
        self.sess.close()
        