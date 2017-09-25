import pickle
import os

from data_preprocessor import DataPreprocessor
from utils import get_pkl_file_name

if __name__ == '__main__':

    # prepare pkl data
    preprocessed_data_dir = './data_pkl'
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
        data_preprocessor = DataPreprocessor('dest_route_pred_sample.csv')
        data_preprocessor.process_and_save(save_dir=preprocessed_data_dir)

    # load trn and tst data
    car_id_list = [5, 9, 14, 29, 50, 72, 74, 100]
    proportion_list = [0.2, 0.4, 0.6, 0.8]
    short_term_dest_list = [-1, 5]

    car_id = car_id_list[0]
    proportion = proportion_list[0]
    dest_term = short_term_dest_list[0]

    trn_fname = get_pkl_file_name(car_id, proportion, dest_term, train=True)
    tst_fname = get_pkl_file_name(car_id, proportion, dest_term, train=False)

    trn_data = pickle.load(open(trn_fname, 'rb'))
    tst_data = pickle.load(open(tst_fname, 'rb'))
    
    # model
    # ...
