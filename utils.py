from datetime import datetime, date
from itertools import product
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


def get_pkl_file_name(car_id, prop_or_min, dest_term):
    base_str = '{car_id}_{prop_or_min_name}_{prop_or_min}_y_{dest_type}.p'
    if prop_or_min < 1:
        prop_or_min_name = 'proportion'
        prop_or_min = int(prop_or_min * 100)
        # prop_or_min = int(max(0.2, prop_or_min)) * 100
    else:
        prop_or_min_name = 'minute'
    file_name = base_str.format(
        car_id='VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        prop_or_min_name=prop_or_min_name,
        prop_or_min=prop_or_min,
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


def resize_to_k(path, k):
    """get the latest k points from the given path (np array)"""
    if len(path) < k: 
        diff = k - len(path)
        back_k = np.concatenate([np.tile(path[0], (diff, 1)), path], axis=0)
    else:
        back_k = path[-k:]
    return back_k

def resize_to_1k(path, k):
    """get the latest k points from the given path (np array)"""
    if len(path) < k: 
        diff = k - len(path)
        back_k = np.concatenate([np.tile(path[0], (diff, 1)), path], axis=0)
    else:
        back_k = path[-k:]
    return back_k 


def load_data(fname, max_length):
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

    # print(paths[:10])
    # print(type(paths[0]), len(paths), metas[0])
    # print( [len(p) for p in paths[:10]] )

    paths = [resize_to_1k(p, max_length) for p in paths if len(p) > 0]
    # paths = [p for p in paths if p is not None]
    # print(paths)
    # print(len(paths))
    if len(paths) > 0:
        paths = np.stack(paths, axis=0)
        metas, dests = np.array(metas), np.array(dests)
        return paths, metas, dests, dts, full_paths
    else:
        return None


def unified_latest_seqdata(car_id_list, proportion_list, dest_term,
                           train_ratio=0.8, seq_len=10, data_dir='./data_pkl'):

    path_list, meta_list, dest_list, dts, full_paths = [], [], [], [], []

    for car_id, proportion in product(car_id_list, proportion_list):
        fname = os.path.join(
            data_dir, get_pkl_file_name(car_id, proportion, dest_term))
        data_result = load_data(fname, max_length=seq_len)

        if data_result is not None:
            path, meta, dest, dt, full_path = data_result
            path_list.append(path)
            meta_list.append(meta)
            dest_list.append(dest)
            dts += dt
            full_paths += full_path

    # unified dataset
    paths = np.concatenate(path_list, axis=0)
    metas = np.concatenate(meta_list, axis=0)
    dests = np.concatenate(dest_list, axis=0)

    data_size = len(paths)

    perm_idxs = np.random.permutation(data_size)

    paths = paths[perm_idxs]
    metas = metas[perm_idxs]
    dests = dests[perm_idxs]
    full_paths = [full_paths[i] for i in perm_idxs]
    dts = [dts[i] for i in perm_idxs]

    trn_size = int(data_size * train_ratio)
    path_trn, path_tst = paths[:trn_size], paths[trn_size:data_size]
    meta_trn, meta_tst = metas[:trn_size], metas[trn_size:data_size]
    dest_trn, dest_tst = dests[:trn_size], dests[trn_size:data_size]
    dt_trn, dt_tst = dts[:trn_size], dts[trn_size:data_size]
    full_path_trn, full_path_tst = full_paths[:trn_size], full_paths[trn_size:data_size]

    return (path_trn, meta_trn, dest_trn, dt_trn, full_path_trn,
            path_tst, meta_tst, dest_tst, dt_tst, full_path_tst)


class Recorder(object):

    def __init__(self, fname):
        self.fname = fname
        self.str_to_record = ''

    def append_values(self, values):
        with open(self.fname, 'a') as fout:
            base_str = '{},' * len(values)
            self.str_to_record += base_str.format(*values)

    def next_line(self):
        with open(self.fname, 'a') as fout:
            self.str_to_record += '\n'
            fout.write(self.str_to_record)
            self.str_to_record = ''


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