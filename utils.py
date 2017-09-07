import os
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Path(object):
    def __init__(self, car_id, start_dt):
        self.car_id = car_id
        self.start_dt = start_dt

        # np array of (x,y)
        self.xy_list = []
        # list of linkID
        self.link_id_list = []

    def add_point(self, x, y, link_id):
        self.xy_list.append((x, y))
        self.link_id_list.append(link_id)

    def preprocess_format(self):
        self.xy = np.array(self.xy_list)
        del self.xy_list
        # time...
        # ...


class DataLoader(object):
    def __init__(self, fname, delimiter=','):
        self.data_dir = './data'
        self.data_fname = os.path.join(self.data_dir, fname)
        fname_prefix, __ = fname.split('.')
        pkl_fname = os.path.join(self.data_dir, fname_prefix + '.p')

        if not os.path.exists(pkl_fname):
            print('Creating {} from raw data ...'.format(pkl_fname))
            self._preprocess_to_pkl(pkl_fname, delimiter)
        else:
            print('Loading data from existing pkl file ... ', end='')

        # load data
        self._load_data(pkl_fname)

    def _preprocess_to_pkl(self, pkl_fname, delimiter):

        # load csv
        header = ['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']
        df = pd.read_csv(self.data_fname, header=None,
                         delimiter=delimiter, names=header, low_memory=False, dtype={'link_id': str})

        # parse data
        car_list = []

        prev_car_id, prev_start_dt = '', 0
        for i, row in enumerate(df.itertuples()):
            # filter 
            if len(row) < 6 + 1:  # column_size + index
                print('Pass the {}-th row : '.format(i), row)
                continue

            is_new_car = prev_car_id != row.car_id
            is_new_path = prev_start_dt != row.start_dt

            if is_new_car:
                new_car = []
                car_list.append(new_car)
            if is_new_path:
                new_path = Path(row.car_id, row.start_dt)
                car_list[-1].append(new_path)

            this_path = car_list[-1][-1]
            this_path.add_point(row.x, row.y, row.link_id)

            prev_car_id, prev_start_dt = row.car_id, row.start_dt

            progress_msg = '\r---Progress...{:10}/{}, num_cars={:3}'
            print(progress_msg.format(i + 1, df.shape[0], len(car_list)),
                  end='', flush=True)
        print('')

        for path_list in car_list:
            for path in path_list:
                path.preprocess_format()

        with open(pkl_fname, 'wb') as fout:
            pickle.dump(car_list, fout)

    def _load_data(self, pkl_fname):
        with open(pkl_fname, 'rb') as fin:
            self._raw_data = pickle.load(fin)
        print('Loading finished. len(vehicle) =', len(self._raw_data))

    @property
    def raw_data(self):
        return self._raw_data