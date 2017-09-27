import argparse
from datetime import datetime
from itertools import product
import time
import os

import tensorflow as tf

from data_preprocessor import DataPreprocessor
import graph
from utils import get_pkl_file_name, load_data, visualize_predicted_destination

DATA_DIR = './data_pkl'

FLAGS = tf.app.flags.FLAGS
FLAGS.num_threads = 1


def train(car_id, proportion, dest_term):
    fname_trn = os.path.join(
        DATA_DIR,
        get_pkl_file_name(car_id, proportion, dest_term, train=True))
    fname_tst = os.path.join(
        DATA_DIR,
        get_pkl_file_name(car_id, proportion, dest_term, train=False))

    input_path_trn, meta_trn, dest_trn = load_data(fname_trn)
    input_path_tst, meta_tst, dest_tst = load_data(fname_tst)

    print('-' * 50)
    print('car_id:', car_id)
    print('trn_data_size:', len(input_path_trn))
    print('tst_data_size:', len(input_path_tst))
    print('-' * 50)
    print('experimental setting:')
    print('  - proportion of path: {}%'.format(100 * proportion))
    print('  - destination: ', end='')
    print('the final.' if dest_term == -1 else str(dest_term) + 'min later.')
    print('-'*50)
    

    # model
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        
        path_trn, meta_trn, dest_trn = graph.inputs(input_path_trn, meta_trn, dest_trn, test=False)
        path_tst, meta_tst, dest_tst = graph.inputs(input_path_tst, meta_tst, dest_tst, test=True)

        # Build inference graph
        with tf.variable_scope('inference') as scope:
            pred_trn = graph.inference(path_trn, meta_trn)
            scope.reuse_variables()
            pred_tst = graph.inference(path_tst, meta_tst)

        # Calculate loss.
        loss_trn = graph.loss(pred_trn, dest_trn)
        loss_tst = graph.loss(pred_tst, dest_tst)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = graph.train(loss_trn, global_step)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create the op for initializing variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(init_op)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # And then after everything is built, start the training loop.
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.
                _, loss_value = sess.run([train_op, loss_trn])

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % FLAGS.log_frequency == 0:
                    examples_per_sec = FLAGS.batch_size / duration

                    # Print status to stdout.
                    print(
                        'Step %d: trn_loss = %.4f, tst_loss = %.4f'
                        '(%.3f sec/batch; %.1f examples/sec)'
                        % (step, loss_value, sess.run(loss_tst),
                           duration, examples_per_sec)
                    )

                    # Update the events file.
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Saving')
            saver.save(sess, FLAGS.train_dir, global_step=step)
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        except KeyboardInterrupt:
            print('>> Training stopped by user!')
        finally:
            # VIZ
            paths, preds, dests = sess.run([path_tst, pred_tst, dest_tst])
            print(paths.shape, preds.shape, dests.shape)
            dest_term = 'F' if dest_term == -1 else str(dest_term)
            for i in range(10):
                fname = './trash/{}-{}-{}-{}.png'.format(car_id, proportion, dest_term, i)
                visualize_predicted_destination(
                    paths[i], dests[i], preds[i], fname=fname)

            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main(_):
    # prepare pkl data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if FLAGS.preprocess:
        data_preprocessor = DataPreprocessor('dest_route_pred_sample.csv')
        data_preprocessor.process_and_save(save_dir=DATA_DIR)
    
    #      (1) 주행경로 길이의 평균
    #      (2) 주행 횟수
    #
    #       (1)    (2)
    #   5: 상위권 하위권
    # 100: 상위권 하위권
    #  29: 상위권 하위권
    #  72: 하위권 상위권
    #  50: 하위권 상위권
    #  14:  평균  상위권
    #   9:  평균   평균
    #  74:  평균   평균

    # load trn and tst data
    car_id_list = [5, 9, 14, 29, 50, 72, 74, 100]

    # Model Configuation
    # meta_embedding_dim_list = [(1,10,4,2)]  # List of Tuple. holi, weekno, hour, weekday
    # path_embedding_dim_list = [16]
    # n_hidden_list = [(16)] # List of Tuple. e.g. [(128),(128, 256)]
    is_rnn_list = [False, True]
    # cell_type_list = ['rnn', 'lstm']
    # bi_direction_list = [True, False]

    # Experiment Configuration
    proportion_list = [0.2, 0.4, 0.6, 0.8]
    short_term_dest_list = [-1, 5]
    use_meta_path = [(True, True), (True, False), (False, True)]

    for params in product(car_id_list, 
                          is_rnn_list,
                          proportion_list, short_term_dest_list, use_meta_path):

        car_id = params[0]

        FLAGS.is_rnn = params[1]
        if FLAGS.is_rnn:
            FLAGS.RNN_architecture = dict()
        else:
            FLAGS.ANN_architecture = dict()

        proportion = params[2]
        dest_term = params[3] 
        
        model_type = 'RNN' if FLAGS.is_rnn else 'ANN'
        
        use_meta = 'META' if params[4][0] else ''
        use_path = 'PATH' if params[4][1] else ''
        FLAGS.feature_set = use_meta + use_path
        
        # Directory where to write event logs and checkpoint.
        if FLAGS.feature_set is 'META':
            FLAGS.train_dir = './tf_logs/VIN_{:03}_{}_{}_y{}/'.format(car_id, FLAGS.feature_set,
                                                              proportion, dest_term)
        else:
            FLAGS.train_dir = './tf_logs/VIN_{:03}_{}_{}_{}_y{}/'.format(car_id, FLAGS.feature_set,
                                                          model_type, proportion, dest_term)
        # train
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
            
        tf.gfile.MakeDirs(FLAGS.train_dir)
        train(car_id, proportion, dest_term)

if __name__ == '__main__':
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=bool, default=False,
                        help='Preprocess data or not')
    parser.add_argument('--gpu_no', type=str, default='-1',
                        help='gpu device number')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--log_freq', type=int, default=1,
                        help='log frequency')

    args, unparsed = parser.parse_known_args()

    # Preprocess data
    FLAGS.preprocess = args.preprocess

    # Data Type
    FLAGS.batch_size = 48
    FLAGS.num_epochs = 3

    # Model Type
    FLAGS.is_rnn = True
    
    # Padding
    FLAGS.k = 10
    
    # GPU setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    
    # Learning params
    FLAGS.lr = args.lr
    FLAGS.log_frequency = args.log_freq
    # Number of batches to run.
    FLAGS.max_steps = 20
    
    tf.app.run()