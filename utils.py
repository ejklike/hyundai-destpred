from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import entropy
import numpy as np


def maybe_exist(directory):
    """make sure the existence of given directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_str_to_time(date_time):
    return datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')


def convert_time_for_fname(date_time):
    if isinstance(date_time, str):
        date_time = convert_str_to_time(date_time)
    return date_time.strftime('%Y%m%d_%H%M%S')


def dist(x, y, to_km=False, std=False):
    rad_to_km = np.array([111.0, 88.8])
    
    x, y = np.array(x).reshape(-1, 2), np.array(y).reshape(-1, 2)
    delta_square = (x - y)**2
    if to_km is True:
        delta_square *= rad_to_km**2
    distances = np.sqrt(np.sum(delta_square, axis=1))
    if std is False:
        return np.mean(distances)
    else:
        return np.mean(distances), np.std(distances)


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    base_str = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'
    file_name = base_str.format(
        train='train' if train else 'test',
        car_id = 'VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion*100) if proportion > 0 else 20,
        dest_type=dest_term)
    return file_name


def load_data(fname, k=0):
  """
  input data:
      paths: list of path nparrays
      metas: list of meta lists
      dests: list of dest nparrays
  output data:
      paths: nparray of shape [data_size, ***]
      metas: nparray of shape [data_size, meta_size]
      dests: nparray of shape [data_size, 2]
  """
  data = pickle.load(open(fname, 'rb'))
  paths, metas, dests = data['path'], data['meta'], data['dest']
  full_paths, dts = data['full_path'], data['dt']

  if k == 0: # RNN or NO PATH
    def resize_by_padding(path, target_length):
      """add zero padding prior to the given path"""
      path_length = path.shape[0]
      pad_width = ((target_length - path_length, 0), (0, 0))
      return np.lib.pad(path, pad_width, 'constant', constant_values=0)
    
    max_length = max(p.shape[0] for p in paths)
    paths = [resize_by_padding(p, max_length) for p in paths]
    paths = np.stack(paths, axis=0)

  else: # DNN
    def resize_to_2k(path, k):
      """remove middle portion of the given path (np array)"""
        # When the prefix of the trajectory contains less than
        # 2k points, the first and last k points overlap
        # (De Brébisson, Alexandre, et al., 2015)
      if len(path) < k: 
        front_k, back_k = np.tile(path[0], (k, 1)), np.tile(path[-1], (k, 1))
      else:
        front_k, back_k = path[:k], path[-k:]
      return np.concatenate([front_k, back_k], axis=0)

    paths = [resize_to_2k(p, k) for p in paths]
    paths = np.stack(paths, axis=0).reshape(-1, 4 * k)
  
  metas, dests = np.array(metas), np.array(dests)

  return paths, metas, dests, dts, full_paths


def record_results(fname, model_id, data_size, global_step, 
                   mean_dist, std_dist, min_dist, max_dist):
    if not os.path.exists(fname):
        with open(fname, 'w') as fout:
            fout.write('model_id,')
            fix = ['trn', 'val', 'tst']
            fout.write(''.join([s + '_size,' for s in fix]))
            fout.write('global_step,')
            fout.write(''.join(['mean_' + s + ',' for s in fix]))
            fout.write(''.join(['std_' + s + ',' for s in fix]))
            fout.write(''.join(['min_' + s + ',' for s in fix]))
            fout.write(''.join(['max_' + s + ',' for s in fix]))                
            fout.write('\n')
    with open(fname, 'a') as fout:
        base_str = '{},' * (2 + 3 * 5) + '\n'
        fout.write(base_str.format(model_id, *data_size, global_step, 
                                   *mean_dist, *std_dist, *min_dist, *max_dist))


def kde_divergence(dest_old, dest_new, bandwidth=0.1):
  kde_old = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_old)
  kde_new = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_new)
  dest_all = np.concatenate([dest_old, dest_new], axis=0)
  score_old = np.exp(kde_old.score_samples(dest_all))
  score_new = np.exp(kde_new.score_samples(dest_all))
  return entropy(score_new, qk=score_old)


def visualize_cluster(dest_trn, dest_val, dest_tst, centers, fname=None, **kwargs):
    if fname is None:
        raise ValueError('You must enter the fname!')

    # data, label, color, marker
    # colorname from https://matplotlib.org/examples/color/named_colors.html
    data_list = [
        ('destinations (trn)', dest_trn, 'lightseagreen', 'o', 0.5),
        ('destinations (val)', dest_val, 'hotpink', '.', 0.5),
        ('destinations (tst)', dest_tst, 'crimson', '.', 0.5),
        ('cluster_centers', centers, 'orangered', '+', 1),
    ]

    fig, ax = plt.subplots()
    for label, path, color, marker, alpha in data_list:
        if path is not None:
            ax.scatter(path[:, 1], path[:, 0], 
                       c=color, marker=marker, label=label, alpha=alpha, s=100)
    ax.legend(); ax.grid(True)

    fname_without_extension = fname[:-4]
    *save_dir, fname_without_dir = fname_without_extension.split('/')
    maybe_exist('/'.join(save_dir))
    car, dest, *_ = fname_without_dir.split('__')
    title = '{car}, {dest}{setting}'.format(
        car=car.upper(),
        dest='FINAL DEST' if dest[-1] == '0' else 'DEST AFTER {} MIN.'.format(dest[-1]),
        setting='' if not kwargs 
                else ('\n(' + ', '.join(['{}={}'.format(k, v) for k, v in kwargs.items()]) + ')'))
        # setting='' if centers is None else '\n(cband=%d, n_centers=%d)'%(cband, len(centers)))
    title += '\n(diag_rad={range_rad:.3f}, diag_km={range_km:.2f}, trn only)'.format(
        range_rad=dist(np.max(dest_trn, axis=0), np.min(dest_trn, axis=0), to_km=False),
        range_km=dist(np.max(dest_trn, axis=0), np.min(dest_trn, axis=0), to_km=True))

    ax_title = ax.set_title(title)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)')
    plt.ylabel('latitude (translated)')
    plt.savefig(fname); plt.close()


def visualize_dest_density(dest_trn, dest_tst, car_id, dest_term,
                           density_segment=10, save_dir='viz'):
    maybe_exist(save_dir)

    #min/max of plot axes
    all_points = np.concatenate([dest_trn, dest_tst], axis=0)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    # add some margin
    dx, dy = xmax-xmin, ymax-ymin
    xmin, xmax = xmin - dx/10, xmax + dx/10
    ymin, ymax = ymin - dy/10, ymax + dy/10
    
    # meshgrid for density scoring
    nx, ny = (100, 100)
    dx, dy = (xmax-xmin)/nx, (ymax-ymin)/ny
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    gridX, gridY = np.meshgrid(xgrid, ygrid) #meshgrid X, Y
        
    # print('score evaluation...')
    gridInput = np.vstack([gridX.ravel(), gridY.ravel()]).T #input 2d vectors for scoring
    bandwidth = dist((xmin, ymin), (xmax, ymax), to_km=False) / 20
    bandwidth_km = dist((xmin, ymin), (xmax, ymax), to_km=True) / 20
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_trn)
    grid_score = np.exp(kde.score_samples(gridInput))
    grid_score = grid_score.reshape(gridX.shape)

    min_score = np.amin(grid_score)
    max_score = np.amax(grid_score)        
    # print('---the sum of score =', dx * dy * np.sum(grid_score))
    # print('---the range of score = ({}, {})'.format(min_score, max_score))

    #plot contour
    fig, ax = plt.subplots()
    ax.set_xlim([ymin, ymax])
    ax.set_ylim([xmin, xmax])
    cmap = plt.get_cmap('Blues')
    levels = np.linspace(min_score, max_score, density_segment)
    cs = ax.contourf(gridY, gridX, grid_score, levels=levels, cmap=cmap, origin='upper')
    fig.colorbar(cs, ax=ax, shrink=0.9)

    #show bandwidth length
    dx, dy = xmax-xmin, ymax-ymin
    lbottom = [xmin + 0.1 * dx, ymin + 0.05 * dy]
    rbottom = [xmin + 0.1 * dx, ymax - 0.05 * dy]
    lscore, rscore = np.exp(kde.score_samples([lbottom, rbottom]))
    if lscore > rscore:
        tx, ty = rbottom
        ty_end, ty_text = ty - bandwidth, ty - 0.9 * bandwidth
    else:
        tx, ty = lbottom
        ty_end, ty_text = ty + bandwidth, ty + 0.1 * bandwidth
    
    ax.plot([ty_end, ty], [tx, tx], 'k-', lw=2)
    ax.annotate('%.2f' % bandwidth, xy=[ty, tx], xytext=[ty_text, tx - 0.05 * dx])

    # data, label, color, marker
    # colorname from https://matplotlib.org/examples/color/named_colors.html
    data_list = [
        ('destinations (trn)', dest_trn, 'slategray', 'o', 0.3),
        ('destinations (tst)', dest_tst, 'orangered', '.', 0.5),
    ]
    for label, path, color, marker, alpha in data_list:
        if path is not None:
            ax.scatter(path[:, 1], path[:, 0], 
                       c=color, marker=marker, label=label, alpha=alpha)
    ax.legend(); ax.grid(True)

    # SET TITLES
    base_title = 'CAR_NO: {car_id}, DEST: {dest}\n(bandwidth_rad={rad:.2f}, bandwidth_km={km:.1f})'
    dest_type = 'FINAL' if dest_term == 0 else 'AFTER_{}_MIN.'.format(dest_term)
    title = base_title.format(car_id=str(car_id).rjust(3, '0'),
                              dest=dest_type,
                              rad=bandwidth, 
                              km=bandwidth_km)
    ax_title = ax.set_title(title, fontsize=12)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)'); plt.ylabel('latitude (translated)')
    
    # SAVE
    fname = os.path.join(save_dir, '{}__CAR{}.png'.format(dest_type, str(car_id).rjust(3, '0')))
    plt.savefig(fname); plt.close()


class DestinationVizualizer(object):

    def __init__(self, path_tst, meta_tst, dest_tst, full_path_tst, dt_tst,
                 full_path_trn, model_id, save_dir='viz'):
        maybe_exist(save_dir)

        self.training_path_points = np.concatenate(full_path_trn, axis=0)
        self.full_path_tst = full_path_tst
        self.dest_tst = dest_tst
        self.dt_tst = dt_tst

        self.meta_tst = meta_tst # or None
        self.path_tst = path_tst
        
        self.model_id = model_id
        self.save_dir = save_dir

    def plot_and_save(self, y_pred, i, **kwargs):
        # data, label, color, marker
        # colorname from https://matplotlib.org/examples/color/named_colors.html
        y_true = self.dest_tst[i]
        full_path = self.full_path_tst[i]
        point_data_list = [
            ('true_destination', y_true, 'mediumblue', '*'),
            ('pred_destination', y_pred, 'crimson', '*'),
        ]
        path_data_list = [ 
            ('full_path', full_path, 'lightgrey', '.'),
        ]

        if self.path_tst is not None:
            input_path = self.path_tst[i]

            if len(input_path.shape) == 1:
                input_path = input_path.reshape(-1, 2) # from 1d to 2d data
                input_path = (input_path[:len(input_path)//2], 
                              input_path[len(input_path)//2:])
                path_data_list += [('model_input', input_path[0], 'mediumblue', '.'),
                                   (None, input_path[1], 'mediumblue', '.')]
            else:
                input_path = input_path[np.sum(input_path, axis=1) != 0, :] # remove zero paddings
                path_data_list += [('model_input', input_path, 'mediumblue', '.')]

        if self.meta_tst is not None:
            holiday, _, hour, weekday = self.meta_tst[i]
            meta_str = '{hour:00} {ampm}, {weekday}{holiday}'.format(
                weekday={0:'MON', 1:'TUE', 2:'WED', 3:'THU', 4:'FRI', 5:'SAT', 6:'SUN'}[weekday], 
                holiday='(h)' if holiday == 1 else '', 
                hour=12 if hour % 12 == 0 else (hour % 12),
                ampm='AM' if hour < 12 else 'PM'
            )
            point_data_list += [(meta_str, [-100, -100], 'white', '')]


def visualize_pred_error(y_true, y_pred, fname=None):
    
    if fname is None:
        raise ValueError('You must enter the fname!')

    y_pred = [x for x in y_pred]
    all_points = np.concatenate([y_true, y_pred], axis=0)

    fig, ax = plt.subplots()
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    dx, dy = 0.1* (xmax - xmin), 0.1 * (ymax- ymin)
    ax.set_xlim([ymin - dy, ymax + dy])
    ax.set_ylim([xmin - dx, xmax + dx])
    
    ax.scatter([x[1] for x in y_true], [x[0] for x in y_true], 
                c='crimson', marker='.', label='true_destination', alpha=0.3, s=50, linewidths=3)
    ax.scatter([x[1] for x in y_pred], [x[0] for x in y_pred], 
               c='mediumblue', marker='.', label='pred_destination', alpha=0.3, s=50, linewidths=3)

    for y1, y2 in zip(y_true, y_pred):
        plt.plot((y1[1], y2[1]), (y1[0], y2[0]), ':', c='xkcd:midnight blue', alpha=0.3)
    ax.legend()
    ax.grid(True)

    # SET TITLE and SAVE
    fname_without_extension = fname[:-4]
    *save_dir, fname_without_dir = fname_without_extension.split('/')
    maybe_exist('/'.join(save_dir))
    car, dest, *exp_setting, _ = fname_without_dir.split('__')
    title = '{car}, {dest}\nSETTING: {setting}'.format(
        car=car.upper(),
        dest='FINAL DEST' if dest[-1] == '0' else 'AFTER {} MIN.'.format(dest[-1]),
        setting='__'.join(exp_setting))
    title += '\n(dist_mean: %.3f (%.3f) rad, %.2f (%.2f) km '%(*dist(y_true, y_pred, to_km=False, std=True), *dist(y_true, y_pred, to_km=True, std=True))
    ax_title = ax.set_title(title)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)')
    plt.ylabel('latitude (translated)')
    plt.savefig(fname)
    plt.close()
