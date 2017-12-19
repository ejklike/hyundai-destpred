"""
python main.py dnn --gpu_no=0 --dest_type=5 --train --record
"""

import argparse
from itertools import product
import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from learner import Model
from utils import (unified_latest_seqdata,
                   Recorder,
                   ResultPlot,
                   dist,
                   convert_time_for_fname
                  )

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = './tf_models'
VIZ_DIR = './viz'

RAW_DATA_FNAME_LIST = ['dest_route_pred_sample.csv', 'dest_route_pred_sample_ag.csv']
RECORD_FNAME = 'result_of_short_term_dest_pred.csv'
N_SPLIT = 5 # for random sampling (select points where predictions occured among a full paths)

proportion_list = [x/N_SPLIT for x in range(1, N_SPLIT) if x/N_SPLIT <= 0.8 and x/N_SPLIT >= 0.2]
print(proportion_list)
car_id_list = [
    'KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
    19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
    39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 
    81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,
] # all

# how-to-suppress-verbose-tensorflow-logging
# https://stackoverflow.com/questions/38073432/
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# FLAGS to be applied across related modules
FLAGS = tf.flags.FLAGS


def train_eval_save(car_id_list, dest_term, model_id, n_save_viz=0):
  """
  TRAIN and EVAL for given car and experimental settings
  """
  # Load datasets
  path_trn, meta_trn, dest_trn, dt_trn, full_path_trn, \
  path_tst, meta_tst, dest_tst, dt_tst, full_path_tst = \
      unified_latest_seqdata(car_id_list, proportion_list, dest_term,
                             train_ratio=0.8,
                             seq_len=FLAGS.seq_len, 
                             data_dir=DATA_DIR)

  print('trn_data:', path_trn.shape, dest_trn.shape)
  print('tst_data:', path_tst.shape, dest_tst.shape)

  # Define model dir
  model_dir = os.path.join(MODEL_DIR, 
                           'dest_type_%d' % dest_term, 
                           'minibatch',
                           model_id)
  model = Model(model_dir)
  FLAGS.train = FLAGS.train or model.latest_checkpoint is None
  
  # Build graph and initialize all variables
  model.build_graph()
  model.init_or_restore_all_variables(restart=FLAGS.restart)

  # TRAIN PART
  if FLAGS.train:
    # model.print_all_trainable_variables()
    model.train(path_trn, meta_trn, dest_trn)

  # TEST EVALUATION PART
  # FOR TARGETING CARS
  for car_id in car_id_list:
    # LOAD DATA
    path_trn, meta_trn, dest_trn, dt_trn, full_path_trn, \
    path_tst, meta_tst, dest_tst, dt_tst, full_path_tst = \
        unified_latest_seqdata([car_id], proportion_list, dest_term,
                                train_ratio=0.8,
                                seq_len=FLAGS.seq_len, 
                                data_dir=DATA_DIR)

    # dist_tst = model.eval_dist(path_tst, meta_tst, dest_tst)
    # recorder = Recorder('PATHWISE_' + RECORD_FNAME)
    # for i in tqdm(range(len(dist_tst))):
    #     recorder.append_values(
    #         ['car{:03}'.format(car_id) if isinstance(car_id, int) else 'car' + car_id,
    #          dt_tst[i], *meta_tst[i], dist_tst[i]])
    #     recorder.next_line()

    if FLAGS.record:
      log.info('save the results to %s', RECORD_FNAME)
      global_step = model.latest_step
      loss_trn = model.eval_mean_distance(path_trn, meta_trn, dest_trn)
      loss_tst = model.eval_mean_distance(path_tst, meta_tst, dest_tst)
      print('car_id:', car_id, 'trn_data:', path_trn.shape, dest_trn.shape, end='--')
      print(loss_trn, loss_tst)

      # SAVE THE RESULT INTO CSV
      recorder = Recorder(RECORD_FNAME)
      recorder.append_values(['car{:03}'.format(car_id) if isinstance(car_id, int) else 'car' + car_id,
                              model_id, 
                              len(path_trn), 
                              len(path_tst), 
                              global_step,
                              loss_trn, 
                              loss_tst])
      recorder.next_line()

    if n_save_viz > 0:
      # DEFINE PLOT AND GET PRED POINTS
      pred_tst = model.predict(path_tst, meta_tst)
      myplot = ResultPlot()
      myplot.add_point(
            path_trn, label=None,
            color='lightgray', marker='.', s=10, alpha=1, must_contain=False)
      myplot.add_point(
            dest_trn, label=None,
            color='gray', marker='.', s=10, alpha=1, must_contain=False)

      # PLOT ALL TEST ERRORS
      for i in range(pred_tst.shape[0]):
        difference = np.stack([dest_tst[i], pred_tst[i]], axis=0)
        myplot.add_tmp_path(
            difference, label=None, 
            color='lightblue', marker=None, must_contain=True)
        myplot.add_tmp_point(
            dest_tst[i], label=None,
            color='mediumblue', marker='*', s=100, alpha=1, must_contain=True)
        myplot.add_tmp_point(
            pred_tst[i], label=None,
            color='crimson', marker='*', s=100, alpha=1, must_contain=True)
      dist_km = dist(dest_tst, pred_tst, to_km=True)

      # Define details to save plot
      save_dir = os.path.join(VIZ_DIR, 
                              'path_and_prediction', 
                              'dest_term_%d' % dest_term, 
                              'car_%03d' % car_id)
      fname = model_id + '.png'
      title = '{fname}\ndist={dist_km}km'
      title = title.format(fname=fname,
                          dist_km='N/A' if dist_km is None else '%.1f' % dist_km)
      myplot.draw_and_save(title, save_dir, fname)

      # FOR EACH TRIP
      for i in range(n_save_viz):
        myplot.add_tmp_path(
              full_path_tst[i], label=None,
              color='lightblue', marker='.', must_contain=True)
        myplot.add_tmp_path(
            path_tst[i], label='input_path', 
            color='mediumblue', marker='.', must_contain=True)

        dest_true, dest_pred = dest_tst[i], pred_tst[i]
        myplot.add_tmp_point(
            dest_true, label='true_destination',
            color='mediumblue', marker='*', s=100, alpha=1, must_contain=True)
        myplot.add_tmp_point(
            dest_pred, label='pred_destination',
            color='crimson', marker='*', s=100, alpha=1, must_contain=True)

        start_time = convert_time_for_fname(dt_tst[i])
        dist_km = dist(dest_pred, dest_true, to_km=True)

        # Define details to save plot
        save_dir = os.path.join(VIZ_DIR, 
                                'path_and_prediction', 
                                'dest_term_%d' % dest_term, 
                                'car_%03d' % car_id, 
                                'start_%s' % start_time)
        fname = model_id + '.png'
        title = '{datetime}\n{fname}\ndist={dist_km}km'
        title = title.format(fname=fname,
                            datetime=start_time,
                            dist_km='N/A' if dist_km is None else '%.1f' % dist_km)
        myplot.draw_and_save(title, save_dir, fname)


  # Close tf session to release GPU memory
  model.close_session()


def main(_):
  """
  MAIN FUNCTION - define loops for experiments
  """
  # Preprocess data: convert to pkl data
  if FLAGS.preprocess:
    for raw_data_fname in RAW_DATA_FNAME_LIST:
      data_preprocessor = DataPreprocessor(to_dir=DATA_DIR)
      data_preprocessor.process_and_save(raw_data_fname)

  # Used for loading data and building graph
  n_hidden_node_list = [100]
  n_hidden_layer_list = [1]

  # PARAM GRIDS
  param_grid_targets = [n_hidden_node_list, # path only
                        n_hidden_layer_list, # for final dense layers
                        ]
  param_product = product(*param_grid_targets)
  print(param_grid_targets)
  param_product_size = np.prod([len(t) for t in param_grid_targets])

  for i, params in enumerate(param_product):
    n_hidden_node, n_hidden_layer = params

    FLAGS.num_units = n_hidden_node
    FLAGS.n_hidden_node = n_hidden_node
    FLAGS.n_hidden_layer = n_hidden_layer

    # Model id
    id_components = [
        '{model}_{edim}x{layer}_last'.format(
            model=('B' if FLAGS.bi_direction else FLAGS.model_type[0].upper()),
            edim=n_hidden_node,
            layer=n_hidden_layer),
        # some details
    ]
    model_id = '__'.join(id_components)

    log.infov('=' * 30 + '{} / {} ({:.1f}%)'.format(
        i + 1, param_product_size, (i + 1) / param_product_size * 100) + '=' * 30)
    log.infov('model_id: ' + model_id)

    train_eval_save(car_id_list, FLAGS.dest_type, model_id, n_save_viz=FLAGS.n_save_viz)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model setting
  parser.add_argument(
      'model_type', 
      type=str, 
      default='dnn',
      help='dnn/rnn')
  parser.add_argument(
      '--bi_direction', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='RNN only, bidirection or not')
  parser.add_argument(
      '--k', 
      type=int, 
      default=5,
      help='parameter k, only for DNN')

  # Data preprocess
  parser.add_argument(
      '--preprocess', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='Preprocess data or not')

  parser.add_argument(
      '--validation_size', 
      type=float, 
      default=0.2,
      help='validation size (default=0.2)')

  # gpu allocation
  parser.add_argument(
      '--gpu_no', 
      type=str, 
      default=None,
      help='gpu device number (must specify to use GPU!)')

  # learning parameters and configs
  parser.add_argument(
      '--lr', 
      type=float, 
      default=0.001,
      help='initial learning rate')
  parser.add_argument(
      '--keep_prob', 
      type=float, 
      default=0.99,
      help='keep_prob for dropout')
  parser.add_argument(
      '--reg_scale', 
      type=float, 
      default=0.01,
      help='scale of regularizer for dense layers')
  parser.add_argument(
      '--steps', 
      type=int, 
      default=10000,
      help='step size')
  parser.add_argument(
      '--log_freq', 
      type=int, 
      default=10,
      help='log frequency')
  parser.add_argument(
      '--early_stopping_rounds', 
      type=int, 
      default=100,
      help='early_stopping_steps = (early_stopping_rounds) * (log frequency)')
  parser.add_argument(
      '--train', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='train or just eval')
  parser.add_argument(
      '--restart', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='delete checkpoint of prev model')
  parser.add_argument(
      '--record', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='save record or not')
  parser.add_argument(
      '--n_save_viz', 
      type=int, 
      default=0,
      help='save "n" viz pngs for the test results')

  parser.add_argument(
      '--dest_type', 
      type=int, 
      default=None)

  # Parse input arguments
  args, unparsed = parser.parse_known_args()

  # GPU PARAMS
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no or '-1'
  FLAGS.gpu_allow_growth = True

  # MODEL PARAMS
  FLAGS.model_type = args.model_type
  FLAGS.dest_type = args.dest_type
  FLAGS.bi_direction = args.bi_direction # RNN only
  FLAGS.seq_len = 10

  # HYPER PARAMS
  FLAGS.learning_rate = args.lr
  FLAGS.keep_prob = args.keep_prob
  FLAGS.reg_scale = args.reg_scale

  # LOOPING PARAMS
  FLAGS.steps = args.steps
  FLAGS.log_freq = args.log_freq
  FLAGS.early_stopping_rounds = args.early_stopping_rounds

  # CONTROL PARAMS
  FLAGS.train = args.train
  FLAGS.restart = args.restart
  FLAGS.record = args.record
  FLAGS.n_save_viz = args.n_save_viz

  #
  FLAGS.preprocess = args.preprocess
  FLAGS.validation_size = args.validation_size

  tf.app.run(main=main)