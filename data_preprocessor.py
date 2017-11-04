import codecs
from datetime import datetime, date
from itertools import product
import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_pkl_file_name


def global_latitude_convertor(x):
    return (x + 600) / (-19)


def global_longitude_convertor(y):
    return (y - 660) / (5)


class MetaHandler(object):

    def __init__(self, event_data_fname):
        self.holiday_set = self._load_holiday_set(event_data_fname)
        
    def _load_holiday_set(self, fname):
        if fname == '-':
            fp = codecs.getreader('utf-8')(sys.stdin)
        else:
            fp = codecs.open(fname, 'rb', encoding='utf-8')
        lines = fp.read()
        fp.close()

        jdata = json.loads(lines)
        
        holiday_set = set()
        for days in jdata['results']:
            holiday_type = days['type']
            # h: 법정공휴일, i: 대체 공휴일
            if (holiday_type == 'h') or (holiday_type == 'i'):
                year, month, day = int(days['year']), int(days['month']), int(days['day'])
                holiday_set.add(date(year, month, day))
        return holiday_set

    def _event_chk(self, start_dt):
        return int(start_dt.date() in self.holiday_set)

    def _hour_chk(self, start_dt):
        return start_dt.hour

    def _week_chk(self, start_dt):
        return start_dt.weekday()
    
    def _week_num_chk(self, start_dt):
        # Monday : 0 ~ Sunday : 6
        return start_dt.isocalendar()[1]
    
    def convert_to_meta(self, start_dt):
        """
        input: start_dt ; 2006-01-10 13:56:16
        output: meta data list [공휴일, 요일, 시간대, 주차]
        """
        start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
        
        return [self._event_chk(start_dt), self._week_num_chk(start_dt),
                self._hour_chk(start_dt), self._week_chk(start_dt)]


class Path(object):

    def __init__(self, car_id, start_dt):
        self.car_id = car_id
        self.start_dt = start_dt
        self.xy_list = []
        self.link_id_list = []
    
    def add_point(self, x, y, link_id):
        self.xy_list.append([x, y])
        self.link_id_list.append(link_id)

    def get_partial_path_and_short_term_dest(self, 
                                             proportion_of_path, 
                                             short_term_pred_min=5):
        path_length = len(self.xy_list)
        partial_path_length = int(path_length * proportion_of_path)

        # "-1" stands for the final destination
        if short_term_pred_min == -1:
            short_term_dest_idx = -1
            is_valid_dest = True
        else:
            short_term_dest_idx = partial_path_length + short_term_pred_min * 2
            is_valid_dest = short_term_dest_idx < path_length

        if is_valid_dest:
            partial_path = np.array(
                self.xy_list[:partial_path_length])
            short_term_dest = np.array(
                self.xy_list[short_term_dest_idx])
            return partial_path, short_term_dest
        else:
            return None

    def get_meta_feature(self, meta_handler):
        return meta_handler.convert_to_meta(self.start_dt)


class DataPreprocessor(object):

    def __init__(self, data_fname):
        self.data_dir = './data'
        self.data_path = os.path.join(self.data_dir, data_fname)
        self._meta_handler = MetaHandler('kor_event_days.json')


    def _load_and_parse_data(self):
            
        header = ['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']
        df = pd.read_csv(self.data_path, header=None,
                           delimiter=',', names=header, low_memory=False, 
                           dtype={'link_id': str})
        
        paths_by_car = dict()

        prev_car_id, prev_start_dt = '', 0
        for i, row in enumerate(df.itertuples()):
            # filter 
            if len(row) < 6 + 1:  # column_size + index
                print('Pass the {}-th row : '.format(i), row)
                continue

            is_new_car = prev_car_id != row.car_id
            is_new_path = prev_start_dt != row.start_dt

            if is_new_car:
                paths_by_car[row.car_id] = list()
            if is_new_path:
                new_path = Path(row.car_id, row.start_dt)
                paths_by_car[row.car_id].append(new_path)

            this_path = paths_by_car[row.car_id][-1]
            # scaling x and y to the original scale of latitude and longitude system
            x = global_latitude_convertor(row.x)
            y = global_longitude_convertor(row.y)
            this_path.add_point(x, y, row.link_id)

            prev_car_id, prev_start_dt = row.car_id, row.start_dt

            progress_msg = '\r---Progress...{:10}/{}, num_cars={:3}'
            print(progress_msg.format(i + 1, df.shape[0], len(paths_by_car)),
                  end='', flush=True)

        print('')
        return paths_by_car


    def process_and_save(self, save_dir):

        print('Starting data preprocessing.')

        # load data parsing results
        tmp_pkl_file = os.path.join(save_dir, 'tmp.p')
        print('Check the existence of {} ...'.format(tmp_pkl_file))
        if not os.path.exists(tmp_pkl_file):
            print('Reading and parsing raw data ...')
            paths_by_car = self._load_and_parse_data()
            pickle.dump(paths_by_car, open(tmp_pkl_file, 'wb'))
            print('Saved to temp pkl file.')
        else:
            print('Use existing pkl file that already parsed.')
            paths_by_car = pickle.load(open(tmp_pkl_file, 'rb'))
            

        proportion_of_path_list = [0.2, 0.4, 0.6, 0.8]
        short_term_pred_min_list = [-1, 5] # -1 for final destination
        train_ratio = 0.8

        # preprocess and save
        for car_id, paths in tqdm(paths_by_car.items()):

            # delete the initial point
            for path in paths:
                path.xy_list.pop(0)

            # exclude too short path
            paths = [path for path in paths if len(path.xy_list) >= 10]

            for proportion, dest_term in product(proportion_of_path_list, 
                                                 short_term_pred_min_list):
                # data to save
                path_list, meta_list, dest_list, dt_list = [], [], [], []

                for path in paths:
                    result = path.get_partial_path_and_short_term_dest(
                        proportion, dest_term)

                    if result is not None:
                        partial_path, short_term_dest = result
                        meta_feature = path.get_meta_feature(self._meta_handler)

                        path_list.append(partial_path)
                        meta_list.append(meta_feature)
                        dest_list.append(short_term_dest)
                        dt_list.append(path.start_dt)
                        # print(car_id, proportion, dest_term)
                        # print(path.xy_list, partial_path, short_term_dest)

                # split data into train and test
                data_size = len(path_list)
                train_size = int(train_ratio * data_size)

                trn_data = dict(
                    path=path_list[:train_size],
                    meta=meta_list[:train_size],
                    dest=dest_list[:train_size],
                    dt=dt_list[:train_size],
                )
                tst_data = dict(
                    path=path_list[train_size:],
                    meta=meta_list[train_size:],
                    dest=dest_list[train_size:],
                    dt=dt_list[train_size:],
                )
                
                # save the results
                trn_fname = get_pkl_file_name(car_id, proportion, dest_term, train=True)
                pickle.dump(trn_data, open(os.path.join(save_dir, trn_fname), 'wb'))
                tst_fname = get_pkl_file_name(car_id, proportion, dest_term, train=False)
                pickle.dump(tst_data, open(os.path.join(save_dir, tst_fname), 'wb'))