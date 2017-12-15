import re
import os
import pandas as pd

from learner import Model
from utils import load_data, get_pkl_file_name

MODEL_DIR = './tf_models'
DATA_DIR = 'data_pkl'
RECORD_DIR = './pred_by_dt'

model_names = ['car002__path20.0__cumulall__MD_20x3', 'car003__path20.0__cumulall__MD_50x2',
               'car004__path20.0__cumulall__MD_50x1', 'car005__path20.0__cumulall__MD_20x1',
               'car007__path20.0__cumulall__MD_20x2', 'car008__path20.0__cumulall__MD_50x1',
               'car011__path20.0__cumulall__MD_50x3', 'car012__path20.0__cumulall__MD_20x2',
               'car013__path20.0__cumulall__MD_100x2', 'car014__path20.0__cumulall__MD_100x2',
               'car016__path20.0__cumulall__MD_50x1', 'car017__path20.0__cumulall__MD_20x1',
               'car018__path20.0__cumulall__MD_100x1', 'car020__path20.0__cumulall__MD_50x3',
               'car021__path20.0__cumulall__MD_100x2', 'car024__path20.0__cumulall__MD_20x3',
               'car025__path20.0__cumulall__MD_20x2', 'car026__path20.0__cumulall__MD_50x1',
               'car028__path20.0__cumulall__MD_50x3', 'car029__path20.0__cumulall__MD_20x2',
               'car030__path20.0__cumulall__MD_100x3', 'car031__path20.0__cumulall__MD_20x2',
               'car032__path20.0__cumulall__MD_100x3', 'car033__path20.0__cumulall__MD_50x2',
               'car034__path20.0__cumulall__MD_100x3', 'car036__path20.0__cumulall__MD_100x3',
               'car042__path20.0__cumulall__MD_50x1', 'car044__path20.0__cumulall__MD_50x3',
               'car049__path20.0__cumulall__MD_50x3', 'car050__path20.0__cumulall__MD_100x2',
               'car052__path20.0__cumulall__MD_100x3', 'car053__path20.0__cumulall__MD_100x1',
               'car054__path20.0__cumulall__MD_20x1', 'car055__path20.0__cumulall__MD_50x2',
               'car056__path20.0__cumulall__MD_100x3', 'car059__path20.0__cumulall__MD_50x1',
               'car060__path20.0__cumulall__MD_50x1', 'car061__path20.0__cumulall__MD_100x2',
               'car062__path20.0__cumulall__MD_100x2', 'car064__path20.0__cumulall__MD_20x1',
               'car065__path20.0__cumulall__MD_100x2', 'car067__path20.0__cumulall__MD_20x2',
               'car070__path20.0__cumulall__MD_100x1', 'car073__path20.0__cumulall__MD_100x3',
               'car074__path20.0__cumulall__MD_50x1', 'car075__path20.0__cumulall__MD_100x2',
               'car076__path20.0__cumulall__MD_100x1', 'car077__path20.0__cumulall__MD_50x3',
               'car078__path20.0__cumulall__MD_20x1', 'car080__path20.0__cumulall__MD_20x1',
               'car081__path20.0__cumulall__MD_20x1', 'car082__path20.0__cumulall__MD_50x1',
               'car083__path20.0__cumulall__MD_50x1', 'car084__path20.0__cumulall__MD_50x3',
               'car089__path20.0__cumulall__MD_50x2', 'car090__path20.0__cumulall__MD_50x2',
               'car091__path20.0__cumulall__MD_20x3', 'car093__path20.0__cumulall__MD_100x3',
               'car098__path20.0__cumulall__MD_20x1', 'car100__path20.0__cumulall__MD_20x3']


def model_name_parser(model_name):
    parsed_list = model_name.split('__')
    dest_term = 0
    car_id = int(re.findall('\d+', parsed_list[0])[0])
    proportion = float(re.findall('\d+', parsed_list[1])[0])
    model_type = 'dnn' if re.findall('[A-Z]+', parsed_list[-1])[0][1] == 'D' else 'rnn'
    model_id = model_name
    n_hidden_layer, n_hidden_node = [int(i) for i in re.findall('\d+', parsed_list[-1])]
    return dest_term, car_id, proportion, model_type, model_id, n_hidden_layer, n_hidden_node


def build_csv_name(car_id):
    return "car_{}_dt_result.csv".format(car_id)


if __name__ == '__main__':
    for model_name in model_names:
        dest_term, car_id, proportion, model_type, model_id, n_hidden_layer, n_hidden_node = model_name_parser(
            model_name)
        # load model
        model_dir = os.path.join(MODEL_DIR,
                                 'dest_type_%d' % dest_term,
                                 'proportion_%.1f' % proportion,
                                 'car_%03d' % car_id,
                                 model_type,
                                 model_id)
        model = Model(model_dir)
        # load data
        fname_trn = os.path.join(
            DATA_DIR,
            get_pkl_file_name(car_id, proportion/100, dest_term, train=True))
        fname_tst = os.path.join(
            DATA_DIR,
            get_pkl_file_name(car_id, proportion/100, dest_term, train=False))
        path_trn, meta_trn, dest_trn, dt_trn, full_path_trn = load_data(fname_trn,
                                                                        k=5,
                                                                        max_length=500)
        path_tst, meta_tst, dest_tst, dt_tst, full_path_tst = load_data(fname_tst,
                                                                        k=5,
                                                                        max_length=500)
        model.prepare_prediction(dest_trn, cband=0.1)
        model.build_graph(learning_rate=0.001, k=5, max_length=500,
                          model_type=model_type, gpu_mem_frac=0.5, gpu_allow_growth=True,
                          keep_prob=0.99, reg_scale=0.01, n_hidden_layer=n_hidden_layer,
                          n_hidden_node=n_hidden_node, use_meta=True, use_path=True,
                          path_embedding_dim=50)
        model.init_or_restore_all_variables(restart=False)
        true, pred = model.get_true_pred(path_tst, meta_tst, dest_tst)

        result_df = pd.DataFrame(data=meta_tst, columns=['event, weekday, hour, week'])
        result_df['result'] = (true == pred)

        result_fname = os.path.join(RECORD_DIR, build_csv_name(car_id))
        result_df.to_csv(result_fname, index=False)
