from data_loader import DataLoader

if __name__ == '__main__':
    data_loader = DataLoader('dest_route_pred_sample.csv')
    data_loader.preprocess_and_save(save_dir='trash')
