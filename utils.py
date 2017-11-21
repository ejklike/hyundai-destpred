from datetime import datetime, date
import os
import pickle

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import entropy
import numpy as np

from log import log


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
    delta_square = (x - y) ** 2
    if to_km is True:
        delta_square *= rad_to_km ** 2
    distances = np.sqrt(np.sum(delta_square, axis=1))
    if std is False:
        return np.mean(distances)
    else:
        return np.mean(distances), np.std(distances)


def get_pkl_file_name(car_id, proportion, dest_term, train=True):
    base_str = '{train}_{car_id}_proportion_{proportion}_y_{dest_type}.p'
    file_name = base_str.format(
        train='train' if train else 'test',
        car_id='VIN_{}'.format(car_id) if isinstance(car_id, int) else car_id,
        proportion=int(proportion * 100) if proportion > 0 else 20,
        dest_type=dest_term)
    return file_name


def resize_by_padding(path, max_full_path):
    """add zero padding after to the given path"""
    path_length = path.shape[0]
    pad_width = ((0, max_full_path - path_length), (0, 0))
    return np.lib.pad(path, pad_width, 'constant', constant_values=0)


def _gen_seq_input(path, max_length, start_index=0, add_eos=False, dtype=np.float32):
    """in: size of [max_length, 2], out: size of [max_length, 3], with adding eos to path
    """
    xy = path[start_index:start_index + max_length]
    eos = np.zeros((xy.shape[0], 1))
    if add_eos:
        eos[-1] = 1.
    seq_input = np.concatenate([xy, eos], axis=1).astype(dtype)
    return resize_by_padding(seq_input, max_length)


def load_seq2seq_data(fname, proportion=None):
    """output: size of [data_size, max_length, 3]
    """
    data = pickle.load(open(fname, 'rb'))
    full_paths, paths, dests, dts = data['full_path'], data['path'], data['dest'], data['dt']

    if proportion is None:
        max_length = max(path.shape[0] for path in full_paths) - 1
    else:
        max_length = max(path.shape[0] for path in paths) - 1
    model_input = [_gen_seq_input(path, max_length, start_index=0, add_eos=False) 
                   for path in full_paths]
    model_output = [_gen_seq_input(path, max_length, start_index=1, add_eos=True) 
                    for path in full_paths]

    model_input = np.stack(model_input, axis=0)
    model_output = np.stack(model_output, axis=0)

    dests = np.array(dests, dtype=np.float32)

    return model_input, model_output, dests, dts

def trim_data(data):
    """remove zero points
    input: [original_length, 2]
    output: [trimmed_length , 2]
    """
    return data[np.sum(data, axis=1) != 0]    

def flat_and_trim_data(data):
    """flatten and remove zero points
    input: [batch_size, seq_length, 3]
    output: [some_length , 2] (some_length < batch_size * seq_length)
    """
    data = data[:, :, :2].reshape(-1, 2)
    return trim_data(data)


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
    ax.legend();
    ax.grid(True)

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
    plt.savefig(fname);
    plt.close()


def visualize_dest_density(dest_trn, dest_tst, car_id, dest_term,
                           density_segment=10, bandwidth_param=20, save_dir='viz'):
    maybe_exist(save_dir)

    # min/max of plot axes
    all_points = np.concatenate([dest_trn, dest_tst], axis=0)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    # add some margin
    dx, dy = xmax - xmin, ymax - ymin
    xmin, xmax = xmin - dx / 10, xmax + dx / 10
    ymin, ymax = ymin - dy / 10, ymax + dy / 10

    # meshgrid for density scoring
    nx, ny = (100, 100)
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    gridX, gridY = np.meshgrid(xgrid, ygrid)  # meshgrid X, Y

    # print('score evaluation...')
    gridInput = np.vstack([gridX.ravel(), gridY.ravel()]).T  # input 2d vectors for scoring
    bandwidth = dist((xmin, ymin), (xmax, ymax), to_km=False) / bandwidth_param
    bandwidth_km = dist((xmin, ymin), (xmax, ymax), to_km=True) / bandwidth_param
    # bandwidth = bandwidth
    # rad_to_km = np.array([111.0, 88.8])
    # bandwidth_km = int(np.sqrt(np.sum(np.array([bandwidth, bandwidth])**2 * rad_to_km**2)))
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dest_trn)
    # kde = BayesianGaussianMixture(n_components=3, weight_concentration_prior=1, weight_concentration_prior_type='dirichlet_distribution').fit(dest_trn)
    # kde = GaussianMixture(n_components=2).fit(dest_trn)
    grid_score = np.exp(kde.score_samples(gridInput))
    grid_score = grid_score.reshape(gridX.shape)

    min_score = np.amin(grid_score)
    max_score = np.amax(grid_score)
    # print('---the sum of score =', dx * dy * np.sum(grid_score))
    # print('---the range of score = ({}, {})'.format(min_score, max_score))

    # plot contour
    fig, ax = plt.subplots()
    ax.set_xlim([ymin, ymax])
    ax.set_ylim([xmin, xmax])
    cmap = plt.get_cmap('Blues')
    levels = np.linspace(min_score, max_score, density_segment)
    cs = ax.contourf(gridY, gridX, grid_score, levels=levels, cmap=cmap, origin='upper')
    fig.colorbar(cs, ax=ax, shrink=0.9)

    # show bandwidth length
    dx, dy = xmax - xmin, ymax - ymin
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
    ax.legend();
    ax.grid(True)

    # SET TITLES
    base_title = 'CAR_NO: {car_id}, DEST: {dest}\n(bandwidth_rad={rad:.2f}, bandwidth_km={km:.1f})'
    dest_type = 'FINAL' if dest_term == 0 else 'AFTER_{}_MIN.'.format(dest_term)
    title = base_title.format(car_id=str(car_id).rjust(3, '0'),
                              dest=dest_type,
                              rad=bandwidth,
                              km=bandwidth_km)
    ax_title = ax.set_title(title, fontsize=12)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)');
    plt.ylabel('latitude (translated)')

    # SAVE
    fname = os.path.join(save_dir, '{}__CAR{}.png'.format(dest_type, str(car_id).rjust(3, '0')))
    plt.savefig(fname)
    plt.close()


class ResultPlot(object):
    def __init__(self, model_id, save_dir='viz'):
        maybe_exist(save_dir)
        self.model_id = model_id
        self.save_dir = save_dir

        # label, data, color, marker, must_contain
        self.path_list = []
        self.point_list = []
        self.lim_list = []

        # temporary memory
        self.tmp_path_list = []
        self.tmp_point_list = []
        self.tmp_lim_list = []

    def add_path(self, data, label=None, 
                 color='black', marker=None, must_contain=False):
        self.path_list.append([label, data, color, marker])
        if must_contain is True:
            self.lim_list.append(data.reshape(-1, 2))

    def add_point(self, data, label=None, 
                  color='black', marker=None, s=100, alpha=1, must_contain=False):
        self.point_list.append([label, data, color, marker, s, alpha])
        if must_contain is True:
            self.lim_list.append(data.reshape(-1, 2))

    def add_tmp_path(self, data, label=None, 
                     color='black', marker=None, must_contain=False):
        self.tmp_path_list.append([label, data, color, marker])
        if must_contain is True:
            self.tmp_lim_list.append(data.reshape(-1, 2))

    def add_tmp_point(self, data, label=None, 
                      color='black', marker=None, s=100, alpha=1, must_contain=False):
        self.tmp_point_list.append([label, data, color, marker, s, alpha])
        if must_contain is True:
            self.tmp_lim_list.append(data.reshape(-1, 2))

    def draw_and_save(self, dist_km=None, **kwargs):
        fig, ax = plt.subplots()

        # concatenate all information
        this_lim_list = self.lim_list + self.tmp_lim_list
        this_point_list = self.point_list + self.tmp_point_list
        this_path_list = self.path_list + self.tmp_path_list

        # set xlim and ylim of viz
        must_visible_points = np.concatenate(this_lim_list, axis=0)
        xmin, ymin = np.min(must_visible_points, axis=0)
        xmax, ymax = np.max(must_visible_points, axis=0)
        dx, dy = 0.1 * (xmax - xmin), 0.1 * (ymax- ymin)
        ax.set_xlim([ymin - dy, ymax + dy])
        ax.set_ylim([xmin - dx, xmax + dx])

        # scatter points
        for label, point, color, marker, s, alpha in this_point_list:
            if np.sum(point.shape) == 2:
                point = point.reshape(-1, 2)
            ax.scatter(point[:, 1], point[:, 0], 
                       c=color, marker=marker, label=label, s=s, alpha=alpha)
        # plot paths
        for label, path, color, marker in this_path_list:
            ax.plot(path[:, 1], path[:, 0], c=color, marker=marker, label=label)
        
        # add regends and grids
        ax.legend(); ax.grid(True)

        # SET TITLES
        kwargs_str = '_'.join([k + '_' + v for k, v in sorted(kwargs.items())])
        title = '{model_id}\n{kwargs_str}, dist={dist_km}km'
        title = title.format(model_id=self.model_id,
                             kwargs_str=kwargs_str,
                             dist_km='N/A' if dist_km is None else '%.1f' % dist_km)
        ax_title = ax.set_title(title, fontsize=12)
        fig.subplots_adjust(top=0.8)
        plt.xlabel('longitude (translated)'); plt.ylabel('latitude (translated)')
        
        # SAVE
        fname = os.path.join(self.save_dir, self.model_id + '__' + kwargs_str + '.png')
        log.info('Save PNG to: %s', fname)
        plt.savefig(fname); plt.close()

        # clean temporary things
        self.tmp_point_list = []
        self.tmp_path_list = []
        self.tmp_lim_list = []


def visualize_pred_error(y_true, y_pred, model_id, save_dir='viz'):
    maybe_exist(save_dir)

    fig, ax = plt.subplots()

    # min/max of plot axes
    all_points = np.concatenate([y_true, y_pred], axis=0)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    dx, dy = 0.1* (xmax - xmin), 0.1 * (ymax- ymin)
    ax.set_xlim([ymin - dy, ymax + dy])
    ax.set_ylim([xmin - dx, xmax + dx])
    
    ax.scatter([x[1] for x in y_true], [x[0] for x in y_true], 
                c='mediumblue', marker='.', label='true_destination', alpha=0.3, s=50, linewidths=3)
    ax.scatter([x[1] for x in y_pred], [x[0] for x in y_pred], 
               c='crimson', marker='.', label='pred_destination', alpha=0.3, s=50, linewidths=3)

    for y1, y2 in zip(y_true, y_pred):
        plt.plot((y1[1], y2[1]), (y1[0], y2[0]), ':', c='xkcd:midnight blue', alpha=0.3)
    ax.legend()
    ax.grid(True)

    # SET TITLES
    title = model_id + '\nError (mean, std) : ({:.2f}, {:.2f})rad, ({:.1f}, {:.1f})km'.format(
        *dist(y_true, y_pred, to_km=False, std=True), *dist(y_true, y_pred, to_km=True, std=True))
    ax_title = ax.set_title(title, fontsize=12)
    fig.subplots_adjust(top=0.8)
    plt.xlabel('longitude (translated)'); plt.ylabel('latitude (translated)')
    
    # SAVE
    fname = os.path.join(save_dir, model_id + '.png')
    plt.savefig(fname); plt.close()
