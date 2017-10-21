import json
from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def preprocess_data(data_dir=None):
    assert data_dir is not None
    # prepare pkl data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    data_preprocessor = DataPreprocessor('dest_route_pred_sample.csv')
    data_preprocessor.process_and_save(save_dir=DATA_DIR)


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    file_name = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'.format(
        train='train' if train else 'test',
        car_id = 'VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion*100),
        dest_type='F' if dest_term == -1 else dest_term)
    return file_name


# def load_data(fname):
#     """for old main module"""
#     data = pickle.load(open(fname, 'rb'))
#     path, meta, dest = data['path'], data['meta'], data['dest']
#     return path, meta, dest


def load_data(fname, k=0):
  """
  input data:
      paths: list of path nparrays
      metas: list of meta lists
      dests: list of dest nparrays
  output data:
      paths: nparray of shape [data_size, ***]
      metas: nparray of shape [data_size, meta_size]
      dests: nparray of shape [data_size, 2]
  """
  data = pickle.load(open(fname, 'rb'))
  paths, metas, dests = data['path'], data['meta'], data['dest']

  if k == 0:
    def resize_by_padding(path, target_length):
      """add zero padding prior to the given path"""
      path_length = path.shape[0]
      pad_width = ((target_length - path_length, 0), (0, 0))
      return np.lib.pad(path, pad_width, 'constant', constant_values=0)
    
    max_length = max(p.shape[0] for p in paths)
    paths = [resize_by_padding(p, max_length) for p in paths]
    paths = np.stack(paths, axis=0)

  else:
    def resize_to_2k(path, k):
      """remove middle portion of the given path (np array)"""
        # When the prefix of the trajectory contains less than
        # 2k points, the first and last k points overlap
        # (De Brébisson, Alexandre, et al., 2015)
      if len(path) < k: 
        front_k, back_k = np.tile(path[0], (k, 1)), np.tile(path[-1], (k, 1))
      else:
        front_k, back_k = path[:k], path[-k:]
      return np.concatenate([front_k, back_k], axis=0)

    paths = [resize_to_2k(p, k) for p in paths]
    paths = np.stack(paths, axis=0).reshape(-1, 4 * k)
  
  metas, dests = np.array(metas), np.array(dests)

  return paths, metas, dests


def record_results(fname, model_id, trn_rmse, tst_rmse):
    if not os.path.exists(fname):
        with open(fname, 'w') as fout:
            fout.write('model_id,trn_rmse,tst_rmse\n')
    with open(fname, 'a') as fout:
        fout.write('{},{},{}\n'.format(model_id, trn_rmse, tst_rmse))


def visualize_path(x, fname=None):
    plt.figure()
    plt.scatter(x[0,0], x[0,1], c='g', marker='o')
    plt.plot(x[:,0], x[:,1], c='g', marker='.')
    plt.scatter(x[-1,0], x[-1,1], c='g', marker='x') # dest
    if fname is None:
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(fname)
    plt.close()


def visualize_predicted_destination(x, y_true, y_pred, fname=None):
    if len(x.shape) == 1:
        # from 1d to 2d data
        path = x.reshape(-1, 2)
    else:
        # remove zero paddings
        path = x[np.sum(x, axis=1) != 0, :]
    # print(path.shape, y_true.shape, y_pred.shape)

    plt.figure()
    plt.scatter(path[0,0], path[0,1], c='g', marker='o')
    plt.plot(path[:,0], path[:,1], c='g', marker='.')
    plt.scatter(y_true[0], y_true[1], c='g', marker='x') # true dest
    plt.scatter(y_pred[0], y_pred[1], c='r', marker='x') # pred dest
    if fname is None:
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(fname)
    plt.close()