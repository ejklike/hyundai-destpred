import os
from datetime import datetime
import numpy as np
import pandas as pd

class Path(object):

    def __init__(self, car_id, start_dt):
 	self.car_id = car_id
	self.start_dt = start_dt

	# np array of (x,y)
	self.xy_list = []
	# timestamp @ departure
	self.time_at_departure = 0
	# list of linkID
	self.link_id_list = []

    def add_point(self, x, y, link_id):
        self.xy_list.append((x,y))
	self.link_id_list.append(link_id)

    def preprocess_format(self):
        self.xy = np.array(self.xy_list)
	del self.xy_list
	# time...
	# ...

class DataLoader(object):

    def __init__(self):
	self.data_dir = './data'
	self.data_fname = os.path.join(self.data_dir, 'data.csv')
	pkl_fname = os.path.join(self.data_dir, 'data.p')

	if not os.path.exists(pkl_fname):
	    print('Creating pkl file from raw data ...')
	    self._preprocess_to_pkl(pkl_fname)

	# load data
	self._load_data(pkl_fname)

    def _preprocess_to_pkl(self, pkl_fname):
        
	# load csv
        header = ['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']
	df = pd.read_csv(self.data_fname, header=None, delimiter=',', names=header)
	
	# parse data
	car_list = []

	prev_car_id, prev_start_dt = '', 0
        for car_id, start_dt, seq_id, x, y, link_id in df.itertuples():
            is_new_car = prev_car_id != car_id
	    is_new_path = prev_start_dt != start_dt

            if is_new_car:
	        new_car = []
	        car_list.append(new_car)
	    if is_new_path:
	        new_path = Path(car_id, start_dt)
	        car_list[-1].append(new_path)
	    
	    this_path = car_list[-1]
	    this_path.add_point(x, y, link_id)
            
	    prev_car_id, prev_start_dt = car_id, start_dt

        for path_list in car_list:
	    for path in path_list:
	        path.preprocess_format()

	with open(pkl_fname, 'wb') as fout:
	    pickle.dump(car_path_list, fout)

    def _load_data(pkl_fname):
	with open(pkl_fname, 'rb') as fin:
	    self.raw_data = pickle.load(fin)
	print('Loading data... len(vehicle) =', len(self.raw_data))


