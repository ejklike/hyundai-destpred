import codecs
import json
from datetime import datetime, date


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