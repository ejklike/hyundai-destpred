from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import entropy

from log import log


def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_str_to_time(date_time):
    return datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')


def convert_time_for_fname(date_time):
    if isinstance(date_time, str):
        date_time = convert_str_to_time(date_time)
    return date_time.strftime('%Y%m%d_%H%M%S')


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    base_str = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'
    file_name = base_str.format(
        train='train' if train else 'test',
        car_id='VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion * 100) if proportion > 0 else 20,
        dest_type=dest_term)
    return file_name


def dist(x, y, to_km=False, std=False):
    rad_to_km = np.array([111.0, 88.8])

    x, y = np.array(x).reshape(-1, 2), np.array(y).reshape(-1, 2)
    delta_square = (x - y) ** 2
    if to_km is True:
        delta_square *= rad_to_km ** 2
    distances = np.sqrt(np.sum(delta_square, axis=1))
    if std is False:
        return np.mean(distances)
    else:
        return np.mean(distances), np.std(distances)


def resize_by_padding(path, target_length):
    """add zero padding AFTER the given path"""
    path_length = path.shape[0]
    pad_width = ((0, target_length - path_length), (0, 0))
    return np.lib.pad(path, pad_width, 'constant', constant_values=0)


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


def trim_data(data):
    """remove zero points
    input: [original_length, 2]
    output: [trimmed_length , 2]
    """
    return data[np.sum(data, axis=1) != 0]    


def flat_and_trim_data(data):
    """flatten and remove zero points
    input: [batch_size, seq_length, 3]
    output: [some_length , 2] (some_length < batch_size * seq_length)
    """
    data = data.reshape(-1, 2)
    return trim_data(data)


def load_data(fname, k=0, max_length=None):
    """
    input data:
        paths: list of path nparrays
        metas: list of meta lists
        dests: list of dest nparrays
    output data:
        paths: nparray of shape [data_size, ***]
        metas: nparray of shape [data_size, meta_size]
        dests: nparray of shape [data_size, 2]
        dts: list of str
        full_paths: list of nparrays
    """
    data = pickle.load(open(fname, 'rb'))
    paths, metas, dests = data['path'], data['meta'], data['dest']
    full_paths, dts = data['full_path'], data['dt']

    if k == 0: # RNN or NO PATH
        paths = [resize_by_padding(p, max_length) for p in paths]
        paths = np.stack(paths, axis=0)

    else: # DNN
        paths = [resize_to_2k(p, k) for p in paths]
        paths = np.stack(paths, axis=0).reshape(-1, 4 * k)
    
    metas, dests = np.array(metas), np.array(dests)

    return paths, metas, dests, dts, full_paths


class Recorder(object):

    def __init__(self, fname):
        self.fname = fname

    def append_values(self, values):
        with open(self.fname, 'a') as fout:
            base_str = '{},' * len(values)
            fout.write(base_str.format(*values))

    def next_line(self):
        with open(self.fname, 'a') as fout:
            fout.write('\n')


        # if not os.path.exists(fname):
        #     with open(fname, 'w') as fout:
        #         # fout.write('model_id,')
        #         # fix = ['trn', 'tst']
        #         # fout.write(''.join([s + '_size,' for s in fix]))
        #         # fout.write('global_step,')
        #         # fout.write(''.join(['mean_' + s + ',' for s in fix]))
        #         # fout.write(''.join(['std_' + s + ',' for s in fix]))
        #         # fout.write(''.join(['min_' + s + ',' for s in fix]))
        #         # fout.write(''.join(['max_' + s + ',' for s in fix]))
        #         fout.write('this is the beginning of recorder')
        #         fout.write('\n')


def kde_divergence(dest_old, dest_new, bandwidth=0.1):
    kde_old = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_old)
    kde_new = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_new)
    dest_all = np.concatenate([dest_old, dest_new], axis=0)
    score_old = np.exp(kde_old.score_samples(dest_all))
    score_new = np.exp(kde_new.score_samples(dest_all))
    return entropy(score_new, qk=score_old)


class ResultPlot(object):
    def __init__(self, save_dir='viz'):
        maybe_exist(save_dir)
        self.save_dir = save_dir

        # label, data, color, marker, must_contain
        self.path_list = []
        self.point_list = []
        self.lim_list = []

        # temporary memory
        self.tmp_path_list = []
        self.tmp_point_list = []
        self.tmp_lim_list = []

    def add_path(self, data, label=None, 
                 color='black', marker=None, must_contain=False):
        self.path_list.append([label, data, color, marker])
        if must_contain is True:
            self.lim_list.append(data.reshape(-1, 2))

    def add_point(self, data, label=None, 
                  color='black', marker=None, s=100, alpha=1, must_contain=False):
        self.point_list.append([label, data, color, marker, s, alpha])
        if must_contain is True:
            self.lim_list.append(data.reshape(-1, 2))

    def add_tmp_path(self, data, label=None, 
                     color='black', marker=None, must_contain=False):
        self.tmp_path_list.append([label, data, color, marker])
        if must_contain is True:
            self.tmp_lim_list.append(data.reshape(-1, 2))

    def add_tmp_point(self, data, label=None, 
                      color='black', marker=None, s=100, alpha=1, must_contain=False):
        self.tmp_point_list.append([label, data, color, marker, s, alpha])
        if must_contain is True:
            self.tmp_lim_list.append(data.reshape(-1, 2))

    def draw_and_save(self, title, save_dir, fname):
        fig, ax = plt.subplots()

        # concatenate all information
        this_lim_list = self.lim_list + self.tmp_lim_list
        this_point_list = self.point_list + self.tmp_point_list
        this_path_list = self.path_list + self.tmp_path_list

        # set xlim and ylim of viz
        must_visible_points = np.concatenate(this_lim_list, axis=0)
        xmin, ymin = np.min(must_visible_points, axis=0)
        xmax, ymax = np.max(must_visible_points, axis=0)
        dx, dy = 0.1 * (xmax - xmin), 0.1 * (ymax- ymin)
        ax.set_xlim([ymin - dy, ymax + dy])
        ax.set_ylim([xmin - dx, xmax + dx])

        # plot paths
        for label, path, color, marker in this_path_list:
            ax.plot(path[:, 1], path[:, 0], c=color, marker=marker, label=label)
        
        # scatter points
        for label, point, color, marker, s, alpha in this_point_list:
            if np.sum(point.shape) == 2:
                point = point.reshape(-1, 2)
            ax.scatter(point[:, 1], point[:, 0], 
                       c=color, marker=marker, label=label, s=s, alpha=alpha)

        # add regends and grids
        ax.legend(); ax.grid(True)

        # SET TITLES
        ax_title = ax.set_title(title, fontsize=12)
        fig.subplots_adjust(top=0.8)
        plt.xlabel('longitude (translated)'); plt.ylabel('latitude (translated)')

        # SAVE
        maybe_exist(save_dir)
        target_path = os.path.join(save_dir, fname)
        log.info('Save PNG to: %s', target_path)
        plt.savefig(target_path); plt.close()

        # clean temporary things
        self.tmp_point_list = []
        self.tmp_path_list = []
        self.tmp_lim_list = []