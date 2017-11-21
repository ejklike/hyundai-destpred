import argparse
from itertools import product
import os

import numpy as np

from data_preprocessor import DataPreprocessor
from log import log
from seq2seq_learner import Model
from utils import (maybe_exist,
                   get_pkl_file_name,
                   flat_and_trim_data,
                   convert_time_for_fname,
                   load_seq2seq_data,
                   dist,
                   ResultPlot)

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = os.path.join(os.getcwd(), './tf_models/seq2seq')
VIZ_DIR = './viz/seq2seq/'

RAW_DATA_FNAME_LIST = ['dest_route_pred_sample.csv', 'dest_route_pred_sample_ag.csv']
RECORD_FNAME = 'result_seq2seq.csv'

dest_term = 0

# how-to-suppress-verbose-tensorflow-logging
# https://stackoverflow.com/questions/38073432/
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def train_eval_save(car_id, proportion, model_id, params):
  """
  TRAIN and EVAL for given car and experimental settings
  """

  # define model dir
  model_dir = os.path.join(MODEL_DIR, model_id)
  model = Model(params, model_dir, restart=FLAGS.restart)

  # Load train dataset
  fname_trn = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=True))
  input_trn, target_trn, _, _ = load_seq2seq_data(fname_trn)
  print('data shape of (trn): ', input_trn.shape)

  # test dataset
  fname_tst = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=False))
  input_tst, _, dest_tst, dt_tst = load_seq2seq_data(fname_tst)
  print('data shape of (tst): ', input_tst.shape)

  # Train Part
  if FLAGS.train:
    model.train(input_trn, target_trn)

  # Test Part
  true_dest, pred_dest = dest_tst, []
  test_input_list = np.split(input_tst, axis=0, indices_or_sections=input_tst.shape[0])
  for i, input_data in enumerate(test_input_list):
    seq_length = int(np.sum(np.sign(np.sum(np.abs(input_data), axis=2))))
    input_data_list = np.split(input_data, input_data.shape[1], axis=1)[:seq_length]
    print('--- test data', i, ', shape of', input_data.shape, ', length:', seq_length)
    # print(input_data[:, :seq_length, :])

    # viz class
    viz_save_dir = os.path.join(VIZ_DIR, 
                                str(car_id), 
                                model_id,
                                str(proportion), 
                                convert_time_for_fname(dt_tst[i]))
    myplot = ResultPlot(model_id, save_dir=viz_save_dir)
    myplot.add_point(
        flat_and_trim_data(input_trn), label=None,
        color='lightgray', marker='.', s=10, alpha=1, must_contain=False)
    myplot.add_point(
        true_dest[i], label='true_destination',
        color='mediumblue', marker='*', s=100, alpha=1, must_contain=True)
    myplot.add_path(
        flat_and_trim_data(input_data), label='input_path', 
        color='mediumblue', marker='.', must_contain=True)

    # input data list for looping RNN
    state_in_v = model.get_test_input_state()
    for counter, prev_input in enumerate(input_data_list[:seq_length], 1):
      next_input, state_in_v = model.predict_next(prev_input, state_in_v)
      dist_km = dist(next_input[:2], true_dest[i], to_km=True)
      print('-input-', next_input[2], next_input[:2], true_dest[i], dist_km)
      myplot.add_tmp_point(
          next_input[:2], label='pred_destination',
          color='crimson', marker='*', s=100, alpha=1, must_contain=True)
      myplot.draw_and_save(dist_km=dist_km, _input=str(counter))

    # predict final destination
    input_v = next_input
    counter = 0
    while input_v[-1] < 0.5 and counter < 30:
      input_v = input_v.reshape((1, 1, 3))
      input_v, state_in_v = model.predict_next(input_v, state_in_v)
      dist_km = dist(input_v[:2], true_dest[i], to_km=True)
      counter += 1
      print('-output-', counter, input_v[2], input_v[:2], true_dest[i], dist_km)
      myplot.add_tmp_point(
          input_v[:2], label='pred_destination',
          color='crimson', marker='*', s=100, alpha=1, must_contain=True)
      myplot.draw_and_save(dist_km=dist_km, _output=str(counter))

    pred_dest.append(next_input[:2])

  model.close_session()


def main():
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
  car_id_list = [FLAGS.car_id] if FLAGS.car_id is not None else ['KMH']

  # input path specification
  proportion_list = [0.4] if FLAGS.proportion is None else [FLAGS.proportion]
  rnn_size_list = [50] if FLAGS.rnn_size is None else [FLAGS.rnn_size]

  # PARAM GRIDS
  param_grid_targets = [car_id_list,
                        proportion_list,
                        rnn_size_list]
  param_product = product(*param_grid_targets)
  print(param_grid_targets)
  param_product_size = np.prod([len(t) for t in param_grid_targets])

  for i, (car_id, proportion, rnn_size) in enumerate(param_product):
    model_params = dict(
        # GPU
        gpu_mem_frac=FLAGS.gpu_mem_frac,
        gpu_allow_growth=False,
        # hyperparameters
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        bernoulli_penalty=FLAGS.bernoulli_penalty,
        validation_size=FLAGS.validation_size,
        early_stopping_rounds=FLAGS.early_stopping_rounds,
        log_freq=FLAGS.log_freq,
        # model type
        n_mixture=FLAGS.n_mixture,
        rnn_size=rnn_size,
    )

    # Model id 'meta' if use_meta else '',
    id_components = [('car{:03}' if isinstance(car_id, int) else 'car{}').format(car_id),
                      'edim{}'.format(rnn_size),
                      'nmix{}'.format(model_params['n_mixture'])]
    model_id = '__'.join(id_components)

    log.infov('=' * 30 + '{} / {} ({:.1f}%)'.format(
        i + 1, param_product_size, (i + 1) / param_product_size * 100) + '=' * 30)
    log.infov('model_id: ' + model_id)
    log.infov('Using params: ' + str(model_params))

    train_eval_save(car_id, proportion, model_id, model_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data preprocess
    parser.add_argument(
        '--preprocess',
        type=bool,
        nargs='?',
        default=False,  # default
        const=True,  # if the arg is given
        help='Preprocess data or not')

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
        '--validation_size',
        type=float,
        default=0.2,
        help='validation size (default=0.2)')
    parser.add_argument(
        '--bernoulli_penalty',
        type=float,
        default=100,
        help='penalty scale for bernoulli NLL')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='batch size')
    parser.add_argument(
        '--steps',
        type=int,
        default=5000,
        help='step size')
    parser.add_argument(
        '--log_freq',
        type=int,
        default=100,
        help='log frequency')
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=5,
        help='early_stopping_steps = (early_stopping_rounds) * (log frequency)')
    parser.add_argument(
        '--train',
        type=bool,
        nargs='?',
        default=False,  # default
        const=True,  # if the arg is given
        help='train or just eval')
    parser.add_argument(
        '--restart',
        type=bool,
        nargs='?',
        default=False,  # default
        const=True,  # if the arg is given
        help='delete checkpoint of prev model')
    # test
    parser.add_argument(
        '--proportion',
        type=float,
        default=None)

    # PARAM GRID args
    parser.add_argument(
        '--car_id',
        type=int,
        default=None)
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=None)
    parser.add_argument(
        '--n_mixture',
        type=int,
        default=20)

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.gpu_no is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main()