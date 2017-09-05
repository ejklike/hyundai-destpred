from utils import DataLoader

def train():
    data_loader = DataLoader('dest_route_pred_sample.csv', delimiter=',')
    for x in data_loader.raw_data[:10]:
        print(x[0].car_id, x[0].start_dt, x[0].xy)
        print(type(x[0].start_dt))

if __name__ == '__main__':
    train()
