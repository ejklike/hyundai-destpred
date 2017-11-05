import json
from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np



def maybe_exist(dir):
    """make sure the existence of given directory"""
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    base_str = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'
    file_name = base_str.format(
        train='train' if train else 'test',
        car_id = 'VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion*100),
        dest_type='F' if dest_term == -1 else dest_term)
    return file_name


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

  if k == 0: # RNN
    def resize_by_padding(path, target_length):
      """add zero padding prior to the given path"""
      path_length = path.shape[0]
      pad_width = ((target_length - path_length, 0), (0, 0))
      return np.lib.pad(path, pad_width, 'constant', constant_values=0)
    
    max_length = max(p.shape[0] for p in paths)
    paths = [resize_by_padding(p, max_length) for p in paths]
    paths = np.stack(paths, axis=0)

  else: # DNN
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


def record_results(fname, model_id, trn_err, val_err, tst_err):
    if not os.path.exists(fname):
        with open(fname, 'w') as fout:
            fout.write('model_id,trn_rmse,tst_rmse\n')
    with open(fname, 'a') as fout:
        fout.write('{},{},{},{}\n'.format(model_id, trn_err, val_err, tst_err))


def visualize_predicted_destination(x, y_true, y_pred, fname=None):
    if len(x.shape) == 1:
        path = x.reshape(-1, 2) # from 1d to 2d data
    else:
        path = x[np.sum(x, axis=1) != 0, :] # remove zero paddings

    # data, label, color, marker
    # colorname from https://matplotlib.org/examples/color/named_colors.html
    path_data_list = [ 
        ('input_path', path, 'mediumblue', '.'),
    ]
    point_data_list = [
        ('starting_point', path[0], 'mediumblue', '.'),
        ('true_destination', y_true, 'mediumblue', '*'),
        ('pred_destination', y_pred, 'crimson', '*'),
    ]

    fig, ax = plt.subplots()
    for label, path, color, marker in path_data_list:
        ax.plot(path[:, 0], path[:, 1], c=color, marker=marker, label=label)
    for label, point, color, marker in point_data_list:
        ax.scatter(point[0], point[1], 
                   c=color, marker=marker, label=label, s=100)#, linewidths=10)
    ax.legend()
    ax.grid(True)

    if fname is None:
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')

    dist_rad = np.sqrt(np.sum((y_true - y_pred)**2))
    dist_km = np.sqrt(np.sum((y_true - y_pred)**2 * [111.0**2, 88.8**2]))
    plt.title(
        fname + '\n(dist_rad={:.3f}, dist_km={:.2f})'.format(dist_rad, dist_km))
    plt.savefig(fname)
    plt.close()