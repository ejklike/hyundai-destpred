import argparse
from itertools import product
import os

import numpy as np

import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from learner import Model
from utils import (get_pkl_file_name, 
                   convert_time_for_fname,
                   load_data,
                   Recorder,
                   ResultPlot,
                   dist,
                   flat_and_trim_data,
                   trim_data,
                  )

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = './tf_models'
VIZ_DIR = './viz'

RAW_DATA_FNAME_LIST = ['dest_route_pred_sample.csv', 'dest_route_pred_sample_ag.csv']
RECORD_FNAME = 'result.csv'

# how-to-suppress-verbose-tensorflow-logging
# https://stackoverflow.com/questions/38073432/
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# FLAGS to be applied across related modules
FLAGS = tf.flags.FLAGS


def train_eval_save(car_id, proportion, dest_term, model_id, n_save_viz=0):
  """
  TRAIN and EVAL for given car and experimental settings
  """
  # Load datasets
  fname_trn = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=True))
  fname_tst = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=False))

  # Numpy Arrays
  path_trn, meta_trn, dest_trn, dt_trn, full_path_trn = load_data(fname_trn, 
                                                                  k=FLAGS.k, 
                                                                  max_length=FLAGS.max_length)
  path_tst, meta_tst, dest_tst, dt_tst, full_path_tst = load_data(fname_tst,
                                                                  k=FLAGS.k, 
                                                                  max_length=FLAGS.max_length)
  print('trn_data:', path_trn.shape, meta_trn.shape, dest_trn.shape)

  # Define model dir
  model_dir = os.path.join(MODEL_DIR, 
                           'dest_type_%d' % dest_term, 
                           'car_{}'.format(car_id),
                           'proportion_%.1f' % proportion, 
                           FLAGS.model_type,
                           model_id)
  model = Model(model_dir)
  FLAGS.train = FLAGS.train or model.latest_checkpoint is None

  # Derive some prerequisites
  # - destination centroids, radius_negibors, ..
  model.prepare_prediction(dest_trn)
  
  # Build graph and initialize all variables
  model.build_graph()
  model.init_or_restore_all_variables(restart=FLAGS.restart)

  # TRAIN PART
  if FLAGS.train:
    # model.print_all_trainable_variables()
    model.train(path_trn, meta_trn, dest_trn)

  # TEST EVALUATION PART
  # Save the test evaluation results
  if FLAGS.record:
    log.info('save the results to %s', RECORD_FNAME)
    global_step = model.latest_step
    loss_m_trn, loss_w_trn = model.eval_metrics(path_trn, meta_trn, dest_trn)
    loss_m_tst, loss_w_tst = model.eval_metrics(path_tst, meta_tst, dest_tst)
    print(loss_m_trn, loss_w_trn)
    print(loss_m_tst, loss_w_tst)
    recorder = Recorder(RECORD_FNAME)
    recorder.append_values([model_id, 
                            len(path_trn), 
                            len(path_tst), 
                            global_step,
                            loss_m_trn, 
                            loss_w_trn, 
                            loss_m_tst, 
                            loss_w_tst])
    recorder.next_line()
  
  # TEST EVALUATION PART
  # Visualize the test evaluation results
  if FLAGS.n_save_viz > 0:
    pred_trn = model.predict(path_trn, meta_trn, dest_trn)
    pred_tst = model.predict(path_tst, meta_tst, dest_tst)

    # Define plot and add training points
    myplot = ResultPlot()
    if proportion > 0:
      myplot.add_point(
            flat_and_trim_data(path_trn), label=None,
            color='lightgray', marker='.', s=10, alpha=1, must_contain=False)
    myplot.add_point(
          dest_trn, label=None,
          color='gray', marker='.', s=10, alpha=1, must_contain=False)
    myplot.add_point(
          model.centroids, label=None,
          color='green', marker='x', s=10, alpha=1, must_contain=False)

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
    
    dist_km, _ = model.eval_metrics(path_tst, meta_tst, dest_tst)
    print('------Error (tst): ', dist_km, 'km')

    # Define details to save plot
    save_dir = os.path.join(VIZ_DIR, 
                            'path_and_prediction', 
                            'dest_term_%d' % dest_term, 
                            'car_{}'.format(car_id))
    fname = model_id[8:] + '.png'
    title = '{fname}\ndist={dist_km}km'
    title = title.format(fname=fname,
                         dist_km='N/A' if dist_km is None else '%.1f' % dist_km)
    myplot.draw_and_save(title, save_dir, fname)

    # Individual visualizations
    for i in range(n_save_viz):
      myplot.add_tmp_path(
            full_path_tst[i], label=None,
            color='lightblue', marker='.', must_contain=True)
      if FLAGS.model_type == 'dnn' and FLAGS.use_path is True:
        input_path1 = path_tst[i, :2*FLAGS.k].reshape(-1, 2)
        input_path2 = path_tst[i, 2*FLAGS.k:].reshape(-1, 2)
        myplot.add_tmp_path(
            input_path1, label='input_path', 
            color='mediumblue', marker='.', must_contain=True)
        myplot.add_tmp_path(
            input_path2, label='input_path', 
            color='mediumblue', marker='.', must_contain=True)
      elif FLAGS.model_type == 'rnn':
        input_path = trim_data(path_tst[i])
        myplot.add_tmp_path(
            input_path, label='input_path', 
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
                              'car_{}'.format(car_id), 
                              'start_%s' % start_time)
      fname = model_id[8:] + '.png'
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

  # training target cars
  # car_id_list = [
  #   'KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
  #   19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
  #   39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
  #   61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 
  #   81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,
  # ] # all
  # car_id_list = {
  #   1:['KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ],
  #   2:[19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, ],
  #   3:[39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, ],
  #   4:[61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, ],
  #   5:[81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,]
  # }[FLAGS.car_group]
#   car_id_list = [5]#, 100, 29, 72, 50, 14, 9, 74] # selected cars
  car_id_list = ['KMH', 5, 100, 29, 72, 50, 15, 9, 74]

  # input path specification
  proportion_list = [0.0, 0.2, 0.4, 0.6, 0.8]

  # Used for loading data and building graph
  use_meta_list = [True, False]
  k_list = [5] if FLAGS.model_type == 'dnn' else [0]
  path_embedding_dim_list = [50] if FLAGS.path_embedding_dim is None else [FLAGS.path_embedding_dim]

  n_hidden_layer_list = [2]

  # PARAM GRIDS
  param_grid_targets = [car_id_list,
                        proportion_list, # path only
                        use_meta_list, # 
                        path_embedding_dim_list, # path only
                        k_list, # path dnn only
                        n_hidden_layer_list, # for final dense layers
                        ]
  param_product = product(*param_grid_targets)
  print(param_grid_targets)
  param_product_size = np.prod([len(t) for t in param_grid_targets])

  for i, params in enumerate(param_product):
    car_id, proportion, use_meta, path_embedding_dim, k, n_hidden_layer = params

    # If we do not use path input,
    # some param grids are not needed.
    if proportion == 0:
      if FLAGS.model_type == 'rnn': # train meta setting only in DNN run
        continue
      if (FLAGS.model_type == 'dnn') and (k != k_list[0]):
        continue
      if use_meta is False:
        continue
      k = 0 # set to 0 after continue statements. this param will be used only for importing data

    # Model id
    id_components = [
        ('car{:03}' if isinstance(car_id, int) else 'car{}').format(car_id),
        'path{:.1f}'.format(proportion),
        '{meta}{model}_{edim}x{layer}'.format(
            meta='M' if use_meta is True else 'X',
            model=('B' if FLAGS.bi_direction else FLAGS.model_type[0].upper()) 
                  if proportion > 0 else '_',
            edim=path_embedding_dim,
            layer=n_hidden_layer),
        'reg_l{}_k{:.2f}'.format(FLAGS.reg_scale, FLAGS.keep_prob),
        'cband_{}'.format(FLAGS.cband),
        # some details
    ]
    model_id = '__'.join(id_components)

    log.infov('=' * 30 + '{} / {} ({:.1f}%)'.format(
        i + 1, param_product_size, (i + 1) / param_product_size * 100) + '=' * 30)
    log.infov('model_id: ' + model_id)

    FLAGS.use_meta = use_meta
    FLAGS.use_path = proportion > 0
    FLAGS.path_embedding_dim = path_embedding_dim
    # FLAGS.k = k
    train_eval_save(car_id, proportion, args.dest_type, 
                    model_id, n_save_viz=FLAGS.n_save_viz)


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
  parser.add_argument(
      '--gpu_mem_frac', 
      type=float, 
      default=1,
      help='use only some portion of the GPU.')
  # Data preprocess
  parser.add_argument(
      '--gpu_allow_growth', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='Allow GPU growth')

  # learning parameters and configs
  parser.add_argument(
      '--learning_rate', 
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
      '--batch_size', 
      type=int, 
      default=1000,
      help='batch size')
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
      default=10,
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

  # PARAM GRID args
  parser.add_argument(
      '--car_group', 
      type=int, 
      default=None)
  parser.add_argument(
      '--proportion', 
      type=float, 
      default=None)

  parser.add_argument(
      '--dest_type', 
      type=int, 
      default=None)
  parser.add_argument(
      '--cband', 
      type=float, 
      default=0)
  parser.add_argument(
      '--radius', 
      type=float, 
      default=5)

  parser.add_argument(
      '--classification', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='save record or not')

  parser.add_argument(
      '--path_dim', 
      type=int, 
      default=None)
  parser.add_argument(
      '--n_dense', 
      type=int, 
      default=None)

  # Parse input arguments
  args, unparsed = parser.parse_known_args()

  # GPU PARAMS
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no or '-1'
  FLAGS.gpu_mem_frac = args.gpu_mem_frac
  FLAGS.gpu_allow_growth = args.gpu_allow_growth

  # MODEL PARAMS
  FLAGS.classification = args.classification
  FLAGS.model_type = args.model_type
  FLAGS.bi_direction = args.bi_direction # RNN only
  FLAGS.max_length= 300 # RNN only
  FLAGS.k = args.k # DNN only
  FLAGS.path_embedding_dim = args.path_dim
  FLAGS.n_hidden_node = args.path_dim
  FLAGS.n_hidden_layer = args.n_dense
  FLAGS.cband = args.cband
  FLAGS.radius = args.radius

  # HYPER PARAMS
  FLAGS.learning_rate = args.learning_rate
  FLAGS.keep_prob = args.keep_prob
  FLAGS.reg_scale = args.reg_scale
  FLAGS.batch_size = args.batch_size

  # LOOPING PARAMS
  FLAGS.steps = args.steps
  FLAGS.log_freq = args.log_freq
  FLAGS.early_stopping_rounds = args.early_stopping_rounds

  # CONTROL PARAMS
  FLAGS.train = args.train
  FLAGS.restart = args.restart
  FLAGS.record = args.record
  FLAGS.n_save_viz = args.n_save_viz
  

  # GRID PARAMS
  # FLAGS.car_group = args.car_group
  # FLAGS.dest_type = args.dest_type
  # car_group, proportion, dest_type, cband, path_dim, n_dense, 
  
  #
  FLAGS.preprocess = args.preprocess
  FLAGS.validation_size = args.validation_size

  tf.app.run(main=main)