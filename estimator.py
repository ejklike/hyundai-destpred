import argparse
from itertools import product
import sys
import os
import pickle

import numpy as np
import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from models import model_fn
from utils import (preprocess_data, 
                   get_pkl_file_name, 
                   load_data, 
                   record_results,
                   visualize_predicted_destination)

FLAGS = None

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = './tf_models'
VIZ_DIR = './viz'
RECORD_FNAME = 'result.csv'

tf.logging.set_verbosity(tf.logging.INFO)


def train_and_eval(car_id, proportion, dest_term, model_id, params):

  # Load datasets
  fname_trn = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=True))
  fname_tst = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=False))

  path_trn, meta_trn, dest_trn = load_data(fname_trn, k=params['k'])
  path_tst, meta_tst, dest_tst = load_data(fname_tst, k=params['k'])
  log.info('trn_data_size, tst_data_size: ' + str(path_trn.shape) + ', ' + str(path_tst.shape))

  # Instantiate Estimator
  model_dir = os.path.join(MODEL_DIR, model_id)
  config = tf.estimator.RunConfig().replace(
      # log_step_count_steps=10,
      save_summary_steps=10,)
  nn = tf.estimator.Estimator(
      model_fn=model_fn,
      params=params,
      config=config,
      model_dir=model_dir)

  # Train
  if FLAGS.train:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'path': path_trn, 'meta': meta_trn},
        y=dest_trn,
        batch_size=path_trn.shape[0],
        num_epochs=None,
        shuffle=True)
    nn.train(input_fn=train_input_fn, steps=FLAGS.steps)

  # Eval
  eval_input_fn_trn = tf.estimator.inputs.numpy_input_fn(
      x={'path': path_trn, 'meta': meta_trn},
      y=dest_trn,
      num_epochs=1,
      shuffle=False)
  eval_input_fn_tst = tf.estimator.inputs.numpy_input_fn(
      x={'path': path_tst, 'meta': meta_tst},
      y=dest_tst,
      num_epochs=1,
      shuffle=False)

  # Scores
  ev_trn = nn.evaluate(input_fn=eval_input_fn_trn)
  ev_tst = nn.evaluate(input_fn=eval_input_fn_tst)
  log.info("Loss: {:.3f}, {:.3f}".format(ev_trn["loss"], ev_tst['loss']))
  log.info("Root Mean Squared Error: {:.3f}, {:.3f}".format(ev_trn["rmse"], ev_tst['rmse']))

  # Viz
  if not os.path.exists(VIZ_DIR):
    os.makedirs(VIZ_DIR)

  pred_tst = nn.predict(input_fn=eval_input_fn_tst)
  import itertools
  pred_tst = itertools.islice(pred_tst, 10)
  for i, pred in enumerate(pred_tst):
    fname = '{}/{}__tst_{}.png'.format(VIZ_DIR, model_id, i)
    visualize_predicted_destination(
        path_tst[i], dest_tst[i], pred, fname=fname)

  return ev_trn['rmse'], ev_tst['rmse']


def main(_):
  # Preprocess data
  if FLAGS.preprocess:
    preprocess_data(data_dir=DATA_DIR)

    # (1) 주행경로 길이의 평균
    # (2) 주행 횟수
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
  proportion_list = [0.2, 0.4, 0.6, 0.8]
  short_term_dest_list = [-1]#, 5]
  use_meta_path = [(True, True), (True, False), (False, True)]

  if FLAGS.model_type == 'dnn':
    k_list = [5, 10, 15, 20]
    bi_direction_list = [False]
  else:
    k_list = [0]
    bi_direction_list = [True, False]

  param_product = product(car_id_list, 
                          proportion_list, 
                          short_term_dest_list, 
                          use_meta_path,
                          k_list,
                          bi_direction_list)

  for params in param_product:
    car_id, proportion, dest_term = params[:3]
    use_meta, use_path = params[3]
    k, bi_direction = params[4:]
    
    path_embedding_dim = 16 if use_path else 0
    n_hidden_layer=1

    model_params = dict(
        learning_rate=FLAGS.learning_rate,
        # feature_set
        use_meta=use_meta,
        use_path=use_path,
        # model type
        model_type=FLAGS.model_type,
        # rnn params
        bi_direction=bi_direction,
        # dnn params
        k=k,
        # path embedding dim (rnn: n_unit / dnn: out_dim)
        path_embedding_dim=path_embedding_dim,
        # the num of final dense layers
        n_hidden_layer=n_hidden_layer,
    )

    # Model id
    model_id = 'car_{:03}'.format(car_id)
    model_id += '__prop_{}'.format(proportion)
    model_id += '__dest_{}'.format(dest_term)
    model_id += '__' + ''.join(['meta' if use_meta else '____', 
                                'path' if use_path else '____'])
    model_id += '__' + ''.join(['b' if bi_direction else '', 
                                FLAGS.model_type])
    model_id += '__k_{}_pdim_{}_dense_{}'.format(k, 
                                                 path_embedding_dim, 
                                                 n_hidden_layer)

    # TODO: some params shoud be added...
    log.warning('=' * 50)
    log.warning('model_id: ' + model_id)
    log.warning('Using params: ' + str(model_params))

    trn_rmse, tst_rmse = train_and_eval(car_id, proportion, dest_term, 
                                        model_id, model_params)
    if FLAGS.train:
      record_results(RECORD_FNAME, model_id, trn_rmse, tst_rmse)
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model setting
  parser.add_argument(
      'model_type', 
      type=str, 
      default='dnn',
      help='dnn/rnn')

  # Data preprocess
  parser.add_argument(
      '--preprocess', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='Preprocess data or not')

  # gpu allocation
  parser.add_argument(
      '--gpu_no', 
      type=str, 
      default=None,
      help='gpu device number')

  # learning parameters and configs
  parser.add_argument(
      '--learning_rate', 
      type=float, 
      default=0.001,
      help='initial learning rate')
  parser.add_argument(
      '--steps', 
      type=int, 
      default=1000,
      help='step size')
  parser.add_argument(
      '--log_freq', 
      type=int, 
      default=1,
      help='log frequency')
  parser.add_argument(
      '--train', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='train or just eval')

  FLAGS, unparsed = parser.parse_known_args()
  
  if FLAGS.gpu_no is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  
  tf.app.run(main=main)