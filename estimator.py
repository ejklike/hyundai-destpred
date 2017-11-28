import argparse
from itertools import product
import os

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.neighbors import radius_neighbors_graph
import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from models import model_fn
from custom_hook import EarlyStoppingHook
from utils import (maybe_exist,
                   get_pkl_file_name,
                   convert_time_for_fname,
                   load_data,
                   record_results,
                   dist,
                   visualize_cluster,
                   visualize_pred_error,
                   flat_and_trim_data,
                   trim_data,
                   ResultPlot)

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = './tf_models'
VIZ_DIR = './viz'

RAW_DATA_FNAME_LIST = ['dest_route_pred_sample.csv', 'dest_route_pred_sample_ag.csv']
RECORD_FNAME = 'result.csv'

# how-to-suppress-verbose-tensorflow-logging
# https://stackoverflow.com/questions/38073432/
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_eval_save(car_id, proportion, dest_term, 
                    model_id, params, n_save_viz=0):
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

  path_trn, meta_trn, dest_trn, _, fpath_trn = load_data(fname_trn, k=params['k'])
  path_tst, meta_tst, dest_tst, dt_tst, fpath_tst = load_data(fname_tst, k=params['k'])

  # split train set into train/validation sets
  num_trn = int(len(path_trn) * (1 - FLAGS.validation_size))
  path_trn, path_val = path_trn[:num_trn], path_trn[num_trn:]
  meta_trn, meta_val = meta_trn[:num_trn], meta_trn[num_trn:]
  dest_trn, dest_val = dest_trn[:num_trn], dest_trn[num_trn:]

  # data for feeding to the graph
  input_dict_trn, input_dict_val, input_dict_tst = {}, {}, {}
  features_val = {}
  if params['use_meta']:
    features_val['meta'] = meta_val
    input_dict_trn['meta'] = meta_trn
    input_dict_val['meta'] = meta_val
    input_dict_tst['meta'] = meta_tst
  if params['use_path']:
    features_val['path'] = path_val
    input_dict_trn['path'] = path_trn
    input_dict_val['path'] = path_val
    input_dict_tst['path'] = path_tst
  params['features_val'] = features_val
  params['labels_val'] = dest_val
  print('data shape of (trn, val, tst): ', path_trn.shape, path_val.shape, path_tst.shape)

  # clustering destinations
  if params['cluster_bw'] > 0:
    cluster_centers = MeanShift(bandwidth=params['cluster_bw']).fit(dest_trn).cluster_centers_
    n_cluster = len(cluster_centers)
    log.info('#cluster of destination = %d', n_cluster)
    params['cluster_centers'] = cluster_centers
    params['n_clusters'] = n_cluster

    if FLAGS.model_type == 'dnn':
      cluster_fname = '{}/cluster/car_{}__dest_{}__cband_{}.png'.format(
          VIZ_DIR, car_id, dest_term, params['cluster_bw'])
      visualize_cluster(dest_trn, dest_val, dest_tst, cluster_centers, 
                        bandwidth=params['cluster_bw'], n_cluster=n_cluster,
                        fname=cluster_fname)

  # input functions for evaluation
  eval_input_fn_trn = tf.estimator.inputs.numpy_input_fn(
      x=input_dict_trn,
      y=dest_trn,
      batch_size=dest_trn.shape[0],
      num_epochs=1,
      shuffle=False)
  eval_input_fn_val = tf.estimator.inputs.numpy_input_fn(
      x=input_dict_val,
      y=dest_val,
      batch_size=dest_val.shape[0],
      num_epochs=1,
      shuffle=False)
  eval_input_fn_tst = tf.estimator.inputs.numpy_input_fn(
      x=input_dict_tst,
      y=dest_tst,
      batch_size=dest_tst.shape[0],
      num_epochs=1,
      shuffle=False)

  # weights for evaluation
  dest_all = np.concatenate([dest_trn, dest_tst], axis=0)
  #         trn     tst
  # trn | trn_trn trn_tst |
  # tst | tst_trn tst_tst |
  connectivity_matrix = radius_neighbors_graph(dest_all, 5, 
                                               mode='connectivity', include_self=False, 
                                               p=2, metric='wminkowski', 
                                               metric_params={'w': [88.8**2, 111.0**2]}).toarray()
  # get only [trn_tst] part and apply reduce_sum
  weight = np.sum(connectivity_matrix[:len(dest_trn), len(dest_trn):], axis=0)
  params['test_weight'] = weight

  # Instantiate Estimator
  model_dir = os.path.join(MODEL_DIR, model_id)
  sess_config = tf.ConfigProto()
  if FLAGS.gpu_mem_frac < 1:
    sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem_frac
  config = tf.estimator.RunConfig(
      tf_random_seed=42,
      save_summary_steps=None,
      save_checkpoints_steps=None,
      save_checkpoints_secs=None,
      session_config=sess_config,
      keep_checkpoint_max=1,
      log_step_count_steps=FLAGS.log_freq)
  nn = tf.estimator.Estimator(
      model_fn=model_fn,
      params=params,
      config=config,
      model_dir=model_dir)

  # check previous ckpt
  ckpt_path = tf.train.latest_checkpoint(model_dir, latest_filename=None)
  print('There exists previously trained model: ', ckpt_path)

  # Train Part
  if FLAGS.train or not ckpt_path:

    # Remove prev model or not
    if FLAGS.restart and tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)

    # Generate infinitely looping batch
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=input_dict_trn,
        y=dest_trn,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)

    # Train
    early_stopping_hook = EarlyStoppingHook(log_freq=FLAGS.log_freq, 
                                            early_stopping_rounds=FLAGS.early_stopping_rounds,
                                            checkpoint_dir=model_dir)
    nn.train(input_fn=train_input_fn, 
             steps=FLAGS.steps, 
             hooks=[early_stopping_hook])

    # check new ckpt
    ckpt_path = tf.train.latest_checkpoint(model_dir, latest_filename=None)
    print('The new model is saved:', ckpt_path)

  # Score evaluation Part
  eval_results = [
      # nn.evaluate(input_fn=eval_input_fn_trn, checkpoint_path=ckpt_path, name='trn'),
      # nn.evaluate(input_fn=eval_input_fn_val, checkpoint_path=ckpt_path, name='val'),
      nn.evaluate(input_fn=eval_input_fn_tst, checkpoint_path=ckpt_path, name='tst')
  ]
  print('eval finished.')
  global_step = eval_results[0]['global_step'] - 1
  # trn_err, val_err, tst_err = [result['mean_distance'] for result in eval_results]
  # trn_werr, val_werr, tst_werr = [result['wmean_distance'] for result in eval_results]
  print('mean(error)', [result['mean_distance'] for result in eval_results])
  print('weighted_mean(error)', [result['wmean_distance'] for result in eval_results])
  # print('std (error)', [result['std_distance'] for result in eval_results])
  # print('min (error)', [result['min_distance'] for result in eval_results])
  # print('max (error)', [result['max_distance'] for result in eval_results])

  # log.warning(model_id)
  # log.warning("Loss {:.3f}, {:.3f}, {:.3f}".format(trn_err, val_err, tst_err))

  if FLAGS.record:
    record_results(RECORD_FNAME, model_id, 
                    data_size=[len(path_trn), len(path_val), len(path_tst)],
                    global_step=global_step, 
                    mean_dist=[result['mean_distance'] for result in eval_results])
                    # wmean_dist=[result['mean_distance'] for result in eval_results],
                    # std_dist=[result['std_distance'] for result in eval_results],
                    # min_dist=[result['min_distance'] for result in eval_results],
                    # max_dist=[result['max_distance'] for result in eval_results])
    log.info('save the results to %s', RECORD_FNAME)

  # PREDICTION
  pred_tst = [x for x in nn.predict(input_fn=eval_input_fn_tst)]

  # Viz Preds
  viz_save_dir = os.path.join(VIZ_DIR, model_id[:23])
  myplot = ResultPlot(model_id, save_dir=viz_save_dir)
  myplot.add_point(
        flat_and_trim_data(path_trn), label=None,
        color='lightgray', marker='.', s=10, alpha=1, must_contain=False)
  myplot.add_point(
        dest_trn, label=None,
        color='lightgray', marker='.', s=10, alpha=1, must_contain=False)

  def _viz_input_and_pred_result(i, desc=None):
    input_path = trim_data(path_tst[i])
    start_time = convert_time_for_fname(dt_tst[i])
    dest_pred = pred_tst[i]
    dest_true = dest_tst[i]
    myplot.add_tmp_path(
        input_path, label='input_path', 
        color='mediumblue', marker='.', must_contain=True)
    myplot.add_tmp_point(
        dest_true, label='true_destination',
        color='mediumblue', marker='*', s=100, alpha=1, must_contain=True)
    myplot.add_tmp_point(
        dest_pred, label='pred_destination',
        color='crimson', marker='*', s=100, alpha=1, must_contain=True)
    dist_km = dist(dest_pred, dest_true, to_km=True)
    if desc is None:
      myplot.draw_and_save(dist_km=dist_km, _datetime=start_time)
    else:
      myplot.draw_and_save(dist_km=dist_km, _datetime=start_time, desc=desc)

  # # plot n_save_viz paths
  # for i in range(n_save_viz):
  #   _viz_input_and_pred_result(i)
  
  # plot argmin/argmax of error
  # tst_argmin = eval_results[2]['argmin_index']
  # tst_argmax = eval_results[2]['argmax_index']
  # print(tst_argmin, tst_argmax)
  # _viz_input_and_pred_result(tst_argmin, desc='best')
  # _viz_input_and_pred_result(tst_argmax, desc='worst')

  # # Plot all (true, pred) destination in test set
  # visualize_pred_error(dest_tst, pred_tst, model_id, save_dir='viz/dest_err_all')


def main(_):
  """
  MAIN FUNCTION - define loops for experiments
  """
  # Preprocess data: convert to pkl data
  if FLAGS.preprocess:
    maybe_exist(DATA_DIR)
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
  car_id_list = {
    1:['KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ],
    2:[19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, ],
    3:[39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, ],
    4:[61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, ],
    5:[81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,]
  }[FLAGS.car_group]
  car_id_list = [5]#, 100, 29, 72, 50, 14, 9, 74] # selected cars

  # input path specification
  short_term_dest_list = [0, 5] if FLAGS.dest_type is None else [FLAGS.dest_type]
  proportion_list = [0.0, 0.2, 0.4, 0.6, 0.8] if FLAGS.proportion is None else [FLAGS.proportion]

  # Used for loading data and building graph
  use_meta_list = [True, False]
  k_list = [5] if FLAGS.model_type == 'dnn' else [0]
  path_embedding_dim_list = [FLAGS.path_dim] if FLAGS.path_dim is not None else [50]

  n_hidden_layer_list = [FLAGS.n_dense] if FLAGS.n_dense is not None else [2]
  cluster_bw_list = [FLAGS.cband] if FLAGS.cband is not None else [0]

  # PARAM GRIDS
  param_grid_targets = [car_id_list,
                        short_term_dest_list, # destination
                        proportion_list, # path only
                        use_meta_list, # 
                        path_embedding_dim_list, # path only
                        k_list, # path dnn only
                        n_hidden_layer_list, # for final dense layers
                        cluster_bw_list
                        ]
  param_product = product(*param_grid_targets)
  print(param_grid_targets)
  param_product_size = np.prod([len(t) for t in param_grid_targets])

  for i, params in enumerate(param_product):
    car_id, dest_term, proportion, use_meta, path_embedding_dim, k = params[:6]
    n_hidden_layer, cluster_bw = params[6:]

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

    model_params = dict(
        learning_rate=FLAGS.learning_rate,
        keep_prob=FLAGS.keep_prob,
        reg_scale=FLAGS.reg_scale,
        # feature_set
        use_meta=use_meta,
        use_path=True if proportion > 0 else False,
        # model type
        model_type=FLAGS.model_type,
        cluster_bw=cluster_bw,
        # rnn params
        bi_direction=FLAGS.bi_direction,
        # dnn params
        k=k,
        # path embedding dim (rnn: n_unit / dnn: out_dim)
        path_embedding_dim=path_embedding_dim,
        # the num of final dense layers
        n_hidden_layer=n_hidden_layer,
    )

    # Model id
    id_components = [
        ('car{:03}' if isinstance(car_id, int) else 'car{}').format(car_id),
        'dest{:02}'.format(dest_term if dest_term > 0 else 0),
        'path{:.1f}'.format(proportion),
        '{meta}{model}_{edim}x{layer}'.format(
            meta='M' if use_meta is True else '-',
            model=('B' if FLAGS.bi_direction else FLAGS.model_type[0].upper()) 
                  if proportion > 0 else '_',
            edim=path_embedding_dim,
            layer=n_hidden_layer),
        'reg_l{}_k{:.2f}'.format(FLAGS.reg_scale, FLAGS.keep_prob),
        'cband_{}'.format(cluster_bw) if cluster_bw > 0 else '',
        # some details
    ]
    model_id = '__'.join(id_components)

    log.infov('=' * 30 + '{} / {} ({:.1f}%)'.format(
        i + 1, param_product_size, (i + 1) / param_product_size * 100) + '=' * 30)
    log.infov('model_id: ' + model_id)
    log.infov('Using params: ' + str(model_params))

    train_eval_save(car_id, proportion, dest_term, 
                    model_id, model_params, n_save_viz=FLAGS.n_save_viz)



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
      default=None)

  parser.add_argument(
      '--path_dim', 
      type=int, 
      default=None)
  parser.add_argument(
      '--n_dense', 
      type=int, 
      default=None)

  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.gpu_no is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  
  tf.app.run(main=main)