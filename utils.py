import os
from datetime import datetime, date
import pickle
import numpy as np
import pandas as pd


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

    def preprocess_format(self, event_data):
        self.xy = np.array(self.xy_list)
        del self.xy_list
        # dt to meta: [공휴일, 요일, 시간대, 주차]
        self.meta = [_event_chk(event_data, self.start_dt), _week_num_chk(self.start_dt), _hour_chk(self.start_dt),
                     _week_chk(self.start_dt)]

        def _event_chk(dict_data, tgt_dtime):
            # 기념일 종류
            '''
            법정공휴일 : h
            법정기념일 : a
            24절기 : s
            그외 절기 : t
            대중기념일 : p
            대체 공휴일 : i
            기타 : e
            '''

            result = 0  # business days

            t_date, t_time = tgt_dtime.split()
            year, month, day = t_date.split('-')

            year, month, day = int(year), int(month), int(day)
            y_s = str(year)
            if month < 10:
                m_s = '0' + str(month)
            else:
                m_s = str(month)
            if day < 10:
                d_s = '0' + str(day)
            else:
                d_s = str(day)

            # print(dict_data['results'])
            for item in dict_data['results']:
                # print(item)
                if item['year'] == y_s and item['month'] == m_s and item['day'] == d_s:
                    if 'h' in item['type'] or 'i' in item['type']:
                        result = 1  # Holidays!
                        # result.append((item['type'], item['name']))

            return result

        def _hour_chk(tgt_dtime):
            # 2006-01-10 13:56:16
            t_date, t_time = tgt_dtime.split()
            hh, mm, ss = t_time.split(':')
            return int(hh)

        def _week_chk(tgt_dtime):
            # Monday : 0 ~ Sunday : 6
            # 2006-01-10 13:56:16
            t_date, t_time = tgt_dtime.split()
            year, month, day = t_date.split('-')
            day_w = date(int(year), int(month), int(day)).weekday()
            return int(day_w)

        def _week_num_chk(tgt_dtime):
            # Monday : 0 ~ Sunday : 6
            # 2006-01-10 13:56:16
            t_date, t_time = tgt_dtime.split()
            year, month, day = t_date.split('-')
            wnum = date(int(year), int(month), int(day)).isocalendar()[1]
            return wnum


class DataLoader(object):
    def __init__(self, fname, delimiter=','):
        self.data_dir = './data'
        self.data_fname = os.path.join(self.data_dir, fname)
        pkl_fname = os.path.join(self.data_dir, 'data.p')

        # load kor_event_days.json
        self.event_data = _load_jsonfile(os.path.join(self.data_dir, 'kor_event_days.json'))

        if not os.path.exists(pkl_fname):
            print('Creating pkl file from raw data ...')
            self._preprocess_to_pkl(pkl_fname, delimiter)

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
            if len(row) < 6 + 1:  # column_size + index
                print('Pass the {}-th row : '.format(i), row)
                continue

            # car_id, start_dt, seq_id, x, y, link_id = row

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
                path.preprocess_format(self.event_data)

        with open(pkl_fname, 'wb') as fout:
            pickle.dump(car_list, fout)

    def _load_data(self, pkl_fname):
        with open(pkl_fname, 'rb') as fin:
            self.raw_data = pickle.load(fin)
        print('Loading data... len(vehicle) =', len(self.raw_data))


def _load_jsonfile(fname):
    if fname == '-':
        fp = codecs.getreader('utf-8')(sys.stdin)
    else:
        fp = codecs.open(fname, 'rb', encoding='utf-8')
    lines = fp.read()
    fp.close()

    jdata = json.loads(lines)
    return jdata
