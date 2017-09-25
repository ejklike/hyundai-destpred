import codecs
import json
from datetime import datetime, date


class MetaHandler(object):
    
    def __init__(self, event_data_fname):
        self.event_data = self._load_jsonfile(event_data_fname)
        
    def _load_jsonfile(self, fname):
        if fname == '-':
            fp = codecs.getreader('utf-8')(sys.stdin)
        else:
            fp = codecs.open(fname, 'rb', encoding='utf-8')
        lines = fp.read()
        fp.close()

        jdata = json.loads(lines)
        return jdata

    def _event_chk(self, tgt_dtime):
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
        for item in self.event_data['results']:
            # print(item)
            if item['year'] == y_s and item['month'] == m_s and item['day'] == d_s:
                if 'h' in item['type'] or 'i' in item['type']:
                    result = 1  # Holidays!
                    # result.append((item['type'], item['name']))

        return result

    def _hour_chk(self, tgt_dtime):
        # 2006-01-10 13:56:16
        t_date, t_time = tgt_dtime.split()
        hh, mm, ss = t_time.split(':')
        return int(hh)

    def _week_chk(self, tgt_dtime):
        # Monday : 0 ~ Sunday : 6
        # 2006-01-10 13:56:16
        t_date, t_time = tgt_dtime.split()
        year, month, day = t_date.split('-')
        day_w = date(int(year), int(month), int(day)).weekday()
        return int(day_w)

    def _week_num_chk(self, tgt_dtime):
        # Monday : 0 ~ Sunday : 6
        # 2006-01-10 13:56:16
        t_date, t_time = tgt_dtime.split()
        year, month, day = t_date.split('-')
        wnum = date(int(year), int(month), int(day)).isocalendar()[1]
        return wnum
    
    def convert_to_meta(self, start_dt):
        # dt to meta: [공휴일, 요일, 시간대, 주차]
        return [self._event_chk(start_dt), self._week_num_chk(start_dt), \
                self._hour_chk(start_dt), self._week_chk(start_dt)]