from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import entropy
import numpy as np


def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_time_for_fname(date_time):
    if isinstance(date_time, str):
        date_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    return date_time.strftime('%Y%m%d_%H%M%S')


def dist(x, y, to_km=False):
    rad_to_km = np.array([111.0, 88.8])
    
    x, y = x.reshape(-1, 2), y.reshape(-1, 2)
    delta_square = (x - y)**2
    if to_km is True:
        delta_square *= rad_to_km**2
    distances = np.sqrt(np.sum(delta_square, axis=1))
    return np.sum(distances)


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    base_str = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'
    file_name = base_str.format(
        train='train' if train else 'test',
        car_id = 'VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion*100) if proportion > 0 else 20,
        dest_type=dest_term)
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
  paths, metas, dests, dts = data['path'], data['meta'], data['dest'], data['dt']

  if k == 0: # RNN or NO PATH
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

  return paths, metas, dests, dts


def record_results(fname, model_id, trn_size, val_size, tst_size, 
                   global_step, trn_err, val_err, tst_err):
    if not os.path.exists(fname):
        with open(fname, 'w') as fout:
            fout.write('model_id,trn_size,val_size,tst_size,global_step,trn_err,val_err,tst_err\n')
    with open(fname, 'a') as fout:
        fout.write('{},{},{},{},{},{},{},{}\n'
                   .format(model_id, trn_size, val_size, tst_size, 
                           global_step, trn_err, val_err, tst_err))


def kde_divergence(dest_old, dest_new, bandwidth=0.1):
  kde_old = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_old)
  kde_new = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_new)
  dest_all = np.concatenate([dest_old, dest_new], axis=0)
  score_old = np.exp(kde_old.score_samples(dest_all))
  score_new = np.exp(kde_new.score_samples(dest_all))
  return entropy(score_new, qk=score_old)


def visualize_cluster(dest_trn, dest_val, dest_tst, centers, fname=None, **kwargs):
    if fname is None:
        raise ValueError('You must enter the fname!')

    # data, label, color, marker
    # colorname from https://matplotlib.org/examples/color/named_colors.html
    data_list = [
        ('destinations (trn)', dest_trn, 'lightseagreen', 'o', 0.5),
        ('destinations (val)', dest_val, 'hotpink', '.', 0.5),
        ('destinations (tst)', dest_tst, 'crimson', '.', 0.5),
        ('cluster_centers', centers, 'orangered', '+', 1),
    ]

    fig, ax = plt.subplots()
    for label, path, color, marker, alpha in data_list:
        if path is not None:
            ax.scatter(path[:, 1], path[:, 0], 
                       c=color, marker=marker, label=label, alpha=alpha, s=100)
    ax.legend(); ax.grid(True)

    fname_without_extension = fname[:-4]
    *save_dir, fname_without_dir = fname_without_extension.split('/')
    maybe_exist('/'.join(save_dir))
    car, dest, *_ = fname_without_dir.split('__')
    title = '{car}, {dest}{setting}'.format(
        car=car.upper(),
        dest='FINAL DEST' if dest[-1] == '0' else 'DEST AFTER {} MIN.'.format(dest[-1]),
        setting='' if not kwargs 
                else ('\n(' + ', '.join(['{}={}'.format(k, v) for k, v in kwargs.items()]) + ')'))
        # setting='' if centers is None else '\n(cband=%d, n_centers=%d)'%(cband, len(centers)))
    title += '\n(diag_rad={range_rad:.3f}, diag_km={range_km:.2f}, trn only)'.format(
        range_rad=dist(np.max(dest_trn, axis=0), np.min(dest_trn, axis=0), to_km=False),
        range_km=dist(np.max(dest_trn, axis=0), np.min(dest_trn, axis=0), to_km=True))

    ax_title = ax.set_title(title)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)')
    plt.ylabel('latitude (translated)')
    plt.savefig(fname); plt.close()


def visualize_predicted_destination(x, y_true, y_pred, fname=None):
    if fname is None:
        raise ValueError('You must enter the fname!')

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
        ax.plot(path[:, 1], path[:, 0], c=color, marker=marker, label=label)
    for label, point, color, marker in point_data_list:
        ax.scatter(point[1], point[0], 
                   c=color, marker=marker, label=label, s=100)#, linewidths=10)
    ax.legend()
    ax.grid(True)

    # SET TITLE and SAVE
    fname_without_extension = fname[:-4]
    *save_dir, fname_without_dir = fname_without_extension.split('/')
    maybe_exist('/'.join(save_dir))
    car, dest, *exp_setting, start_dt = fname_without_dir.split('__')
    title = '{car}, {start_dt}, {dest}\nSETTING: {setting}'.format(
        car=car.upper(),
        start_dt=start_dt,
        dest='FINAL DEST' if dest[-1] == '0' else 'AFTER {} MIN.'.format(dest[-1]),
        setting='__'.join(exp_setting))
    title += '\n(dist_rad={dist_rad:.3f}, dist_km={dist_km:.2f})'.format(
        dist_rad=dist(y_true, y_pred, to_km=False),
        dist_km=dist(y_true, y_pred, to_km=True))
    ax_title = ax.set_title(title)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)')
    plt.ylabel('latitude (translated)')
    plt.savefig(fname)
    plt.close()