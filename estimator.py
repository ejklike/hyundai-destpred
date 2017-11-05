import argparse
from itertools import islice, product
import sys
import os
import pickle

import numpy as np
from sklearn.cluster import MeanShift
import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from models import model_fn
from utils import (maybe_exist,
                   get_pkl_file_name,
                   load_data,
                   record_results,
                   visualize_predicted_destination)

FLAGS = None

# Data dir
DATA_DIR = './data_pkl'
MODEL_DIR = './tf_models'
VIZ_DIR = './viz'

RAW_DATA_FNAME = 'dest_route_pred_sample.csv'
RECORD_FNAME = 'result.csv'

tf.logging.set_verbosity(tf.logging.INFO)


def train_and_eval(car_id, proportion, dest_term, model_id, params, viz=False):
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

    print(path_trn[0], path_tst[0])

    # data for feeding to the graph
    input_dict_trn, input_dict_tst = {}, {}
    if params['use_meta']:
        input_dict_trn['meta'] = meta_trn
        input_dict_tst['meta'] = meta_tst
    if params['use_path']:
        input_dict_trn['path'] = path_trn
        input_dict_tst['path'] = path_tst
    if params['use_cluster']:
        ms = MeanShift()
        ms.fit(dest_trn)
        input_dict_trn['dest_centroid'] = np.repeat(np.expand_dims(ms.cluster_centers_, axis=0), path_trn.shape[0], axis=0).astype(np.float32)
        input_dict_tst['dest_centroid'] = np.repeat(np.expand_dims(ms.cluster_centers_, axis=0), path_tst.shape[0], axis=0).astype(np.float32)


    # Instantiate Estimator
    model_dir = os.path.join(MODEL_DIR, model_id)
    config = tf.estimator.RunConfig().replace(
        # log_step_count_steps=10,
        save_summary_steps=10, )
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=config,
        model_dir=model_dir)

    # Train
    if FLAGS.train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=input_dict_trn,
            y=dest_trn,
            batch_size=path_trn.shape[0],
            num_epochs=None,
            shuffle=True)
        nn.train(input_fn=train_input_fn, steps=FLAGS.steps)

    # Eval
    eval_input_fn_trn = tf.estimator.inputs.numpy_input_fn(
        x=input_dict_trn,
        y=dest_trn,
        num_epochs=1,
        shuffle=False)
    eval_input_fn_tst = tf.estimator.inputs.numpy_input_fn(
        x=input_dict_tst,
        y=dest_tst,
        num_epochs=1,
        shuffle=False)

    # Scores
    ev_trn = nn.evaluate(input_fn=eval_input_fn_trn)
    ev_tst = nn.evaluate(input_fn=eval_input_fn_tst)
    log.info("Loss (trn, tst): {:.3f}, {:.3f}".format(ev_trn["loss"], ev_tst['loss']))

    # Viz
    if viz:
        maybe_exist(VIZ_DIR)
        pred_tst = nn.predict(input_fn=eval_input_fn_tst)
        pred_tst = islice(pred_tst, 10)
        for i, pred in enumerate(pred_tst):
            fname = '{}/{}__tst_{}.png'.format(VIZ_DIR, model_id, i)
            visualize_predicted_destination(
                path_tst[i], dest_tst[i], pred, fname=fname)

    return ev_trn['loss'], ev_tst['loss']


def main(_):
    # Preprocess data: convert to pkl data
    if FLAGS.preprocess:
        maybe_exist(DATA_DIR)
        data_preprocessor = DataPreprocessor(RAW_DATA_FNAME)
        data_preprocessor.process_and_save(save_dir=DATA_DIR)

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

    # input specification
    car_id_list = [100]  # [5, 9, 14, 29, 50, 72, 74, 100]
    proportion_list = [0.4]  # [0.2, 0.4, 0.6, 0.8]
    short_term_dest_list = [5]  # [-1, 5]
    use_cluster_dest = [False, True]

    # Used for loading data and building graph
    use_meta_path = [(False, True)]  # , (True, False), (False, True)]
    if FLAGS.model_type == 'dnn':
        k_list = [5]  # , 10, 15, 20]
        bi_direction_list = [False]
    else:
        k_list = [0]
        bi_direction_list = [True, False]

    path_embedding_dim_list = [16]
    n_hidden_layer_list = [1]

    # PARAM GRIDS
    param_product = product(car_id_list,
                            proportion_list,
                            short_term_dest_list,
                            use_cluster_dest,
                            use_meta_path,
                            k_list,
                            bi_direction_list,
                            path_embedding_dim_list,
                            n_hidden_layer_list)
    param_product_size = np.prod([len(car_id_list),
                                  len(proportion_list),
                                  len(short_term_dest_list),
                                  len(use_cluster_dest),
                                  len(use_meta_path),
                                  len(k_list),
                                  len(bi_direction_list),
                                  len(path_embedding_dim_list),
                                  len(n_hidden_layer_list)])

    for i, params in enumerate(param_product):
        car_id, proportion, dest_term, use_cluster = params[:4]
        use_meta, use_path = params[4]
        k, bi_direction, path_embedding_dim, n_hidden_layer = params[5:]

        # If we do not use path input,
        # some param grids are not needed.
        if use_path is False:
            if proportion > proportion_list[0]:
                continue
            if (FLAGS.model_type == 'dnn') and (k > k_list[0]):
                continue
            # this param is useless
            path_embedding_dim = 0

        model_params = dict(
            learning_rate=FLAGS.learning_rate,
            # feature_set
            use_meta=use_meta,
            use_path=use_path,
            use_cluster=use_cluster,
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
        model_id += '__cluster_{}'.format('T' if use_cluster else 'F')
        model_id += '__' + ''.join(['meta' if use_meta else '____',
                                    'path' if use_path else '____'])
        model_id += '__' + ''.join(['b' if bi_direction else '',
                                    FLAGS.model_type])
        model_id += '__k_{}_pdim_{}_dense_{}'.format(k,
                                                     path_embedding_dim,
                                                     n_hidden_layer)

        log.warning('=' * 30 + '{} / {} ({:.1f}%)'.format(
            i + 1, param_product_size, (i + 1) / param_product_size * 100) + '=' * 30)
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
        default=False,  # default
        const=True,  # if the arg is given
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
        default=False,  # default
        const=True,  # if the arg is given
        help='train or just eval')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.gpu_no is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    tf.app.run(main=main)
