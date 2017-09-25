import os
import pickle
import numpy as np
import pandas as pd
from utils import MetaHandler
from tqdm import tqdm


class Path(object):
    
    def __init__(self, car_id, start_dt):
        self.car_id = car_id
        self.start_dt = start_dt
        self.xy_list = []
        self.link_id_list = []
    
    def add_point(self, x, y, link_id):
        self.xy_list.append((x, y))
        self.link_id_list.append(link_id)
         
    def get_partial_path(self, proportion_of_path):
        path_length = len(self.xy_list)
        return self.xy_list[:int(path_length * proportion_of_path)]

    def get_short_term_destination(self, partial_path, short_term_pred_min=5):
        short_term_destination_index = len(partial_path) + short_term_pred_min * 2
        if short_term_destination_index >= len(self.xy_list):
            return None
        else:
            return self.xy_list[short_term_destination_index]

    def get_meta_feature(self, meta_handler):
        return meta_handler.convert_to_meta(self.start_dt)
    
    def get_final_destination(self):
        return self.xy_list[-1]

    
class DataLoader(object):
    
    def __init__(self, data_fname):
        self.data_dir = './data'
        self.data_path = os.path.join(self.data_dir, data_fname)
        self._meta_handler = MetaHandler(os.path.join(self.data_dir, 'kor_event_days.json'))

    def _load_data(self):
        header = ['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']
        return pd.read_csv(self.data_path, header=None,
                           delimiter=',', names=header, low_memory=False, 
                           dtype={'link_id': str})
        
    def _parse_data(self):

        df = self._load_data()
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
    
    def _save_to_pkl(self, save_dir, car_id, proportion, short_term_pred_min, data):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = '{0}_{1}_y{2}.p'.format(car_id, int(proportion*100), short_term_pred_min)
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(data, f)
    
    def preprocess_and_save(self, save_dir):
        
        proportion_of_path_list = [0.2, 0.4, 0.6, 0.8]
        short_term_pred_min_list = [5]
        
        paths_by_car = self._parse_data()
        
        for car_id, paths in tqdm(paths_by_car.items()):
            
            for path in paths:
                
                meta_feature = path.get_meta_feature(self._meta_handler)
                final_destination = path.get_final_destination()
                
                for proportion in proportion_of_path_list:
                    
                    partial_path = path.get_partial_path(proportion)
                    
                    final_dest_pred_data = {
                        'input':partial_path,
                        'meta':meta_feature,
                        'dest':final_destination
                    }
                    self._save_to_pkl(save_dir, car_id, proportion, 'F', final_dest_pred_data)
                    
                    for short_term_pred_min in short_term_pred_min_list:
                        
                        short_term_destination = path.get_short_term_destination(partial_path, short_term_pred_min)
                        
                        if short_term_destination:
                            short_term_dest_pred_data = {
                                'input':partial_path,
                                'meta':meta_feature,
                                'dest':short_term_destination
                            }
                            self._save_to_pkl(save_dir, car_id, proportion, short_term_pred_min, short_term_dest_pred_data)
