import codecs
import json
from datetime import datetime, date
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

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


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    file_name = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'.format(
        train='train' if train else 'test',
        car_id = 'VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion*100),
        dest_type='F' if dest_term == -1 else dest_term,
    )
    return file_name


def load_data(fname):
    data = pickle.load(open(fname, 'rb'))
    path, meta, dest = data['path'], data['meta'], data['dest']
    return path, meta, dest


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
    # remove zero paddings
    path = x[np.sum(x, axis=1) != 0, :]

    plt.figure()
    plt.scatter(path[0,0], path[0,1], c='g', marker='o')
    plt.plot(path[:,0], path[:,1], c='g', marker='.')
    plt.scatter(y_true[0], y_true[1], c='g', marker='x') # true dest
    plt.scatter(y_pred[0], y_pred[1], c='r', marker='x') # pred dest
    if fname is None:
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(fname)
    plt.close()