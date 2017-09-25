from itertools import product
import os
import pickle

import numpy as np
import pandas as pd
from utils import MetaHandler
from tqdm import tqdm


def save_partial_paths_and_dest(
    paths, proportion, dest_term, save_dir='data_pkl'):
    
    # data to save
    data = dict(
        input=[],
        meta=[],
        dest=[],
    )

    for path in paths:
        # delete initial point
        path.xy_list = path.xy_list[1:]

        # exclude too short path
        if len(path.xy_list) < 10:
            continue

        result = path.get_partial_path_and_short_term_dest(
            proportion, dest_term)

        if result is not None:
            partial_path, short_term_dest = result
            meta_feature = path.get_meta_feature(self._meta_handler)

            data['input'].append(partial_path)
            data['meta'].append(meta_feature)
            data['dest'].append(short_term_dest)

    # save the results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = '{car_id}_proportion_{proportion}_y_{dest_type}.p'.format(
        car_id=car_id,
        proportion=int(proportion*100),
        dest_type='F' if dest_term == -1 else dest_term,
    )
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(data, f)


class Path(object):
    
    def __init__(self, car_id, start_dt):
        self.car_id = car_id
        self.start_dt = start_dt
        self.xy_list = []
        self.link_id_list = []
    
    def add_point(self, x, y, link_id):
        self.xy_list.append((x, y))
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
            partial_path = self.xy_list[:partial_path_length]
            short_term_dest = self.xy_list[short_term_dest_idx]
            return partial_path, short_term_dest
        else:
            return None

    def get_meta_feature(self, meta_handler):
        return meta_handler.convert_to_meta(self.start_dt)


class DataLoader(object):

    def __init__(self, data_fname):
        self.data_dir = './data'
        self.data_path = os.path.join(self.data_dir, data_fname)
        self._meta_handler = MetaHandler(
            os.path.join(self.data_dir, 'kor_event_days.json'))

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
            this_path.add_point(row.x, row.y, row.link_id)

            prev_car_id, prev_start_dt = row.car_id, row.start_dt

            progress_msg = '\r---Progress...{:10}/{}, num_cars={:3}'
            print(progress_msg.format(i + 1, df.shape[0], len(paths_by_car)),
                  end='', flush=True)
        print('')
        return paths_by_car


    def preprocess_and_save(self, save_dir):

        proportion_of_path_list = [0.2, 0.4, 0.6, 0.8]
        short_term_pred_min_list = [-1, 5] # -1 for final destination

        # load data parsing results
        tmp_pkl_file = os.path.join(save_dir, 'tmp.p')
        if not os.path.exists(tmp_pkl_file):
            paths_by_car = self._load_and_parse_data()
            pickle.dump(paths_by_car, open(tmp_pkl_file, 'wb'))
        else:
            paths_by_car = pickle.load(open(tmp_pkl_file, 'rb'))

        # preprocess and save
        for car_id, paths in tqdm(paths_by_car.items()):
            for proportion, dest_term in product(proportion_of_path_list, 
                                                 short_term_pred_min_list):
                save_partial_paths_and_dest(paths, proportion, dest_term)