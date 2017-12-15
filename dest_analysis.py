from itertools import product
import os

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
import tensorflow as tf

from data_preprocessor import DataPreprocessor
from log import log
from utils import (maybe_exist,
                   dist,
                   get_pkl_file_name,
                   load_data,
                   kde_divergence,
                   visualize_dest_density,
                   visualize_pred_error,
                   visualize_pred_error_two_split,
                   visualize_cluster)


# TODO: Quick hack for loading model and inference. (via input_ftn???)
# https://gist.github.com/Inoryy/b606aafea2e3faa3ca847d7be986c999

FLAGS = None

# Data dir
DATA_DIR = './data_pkl'
VIZ_DIR = './viz/dest_analysis'

import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)


# training target cars
car_id_list = ['KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,]#['KMH', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, ] # all
proportion = 0.2
short_term_dest_list = [0]#, 5]
validation_size = 0.2
bandwidth = 0.1

for car_id, dest_term in product(car_id_list, short_term_dest_list):
  # Load datasets
  fname_trn = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=True))
  fname_tst = os.path.join(
      DATA_DIR,
      get_pkl_file_name(car_id, proportion, dest_term, train=False))

  path_trn, meta_trn, dest_trn, _, fpath_trn = load_data(fname_trn, k=0)
  path_tst, meta_tst, dest_tst, _, fpath_tst = load_data(fname_tst, k=0)

  fpath_all = fpath_trn + fpath_tst
  start = [p[0] for p in fpath_all]
  end = [p[-1] for p in fpath_all]

  start = np.stack(start, axis=0)
  end = np.stack(end, axis=0)

  print(car_id, start.shape, end.shape)

  start_trn = [p[0] for p in fpath_trn]
  end_trn = [p[-1] for p in fpath_trn]

  start_trn = np.stack(start_trn, axis=0)
  end_trn = np.stack(end_trn, axis=0)

  start_tst = [p[0] for p in fpath_tst]
  end_tst = [p[-1] for p in fpath_tst]

  start_tst = np.stack(start_tst, axis=0)
  end_tst = np.stack(end_tst, axis=0)

  # visualize_pred_error(start, end, 'Car_{}'.format(car_id), save_dir='viz/start_end')
  visualize_pred_error_two_split(start_trn, end_trn, start_tst, end_tst, 
                                 'TWO_SPLIT_Car_{}'.format(car_id), save_dir='viz/start_end')

#   def count_closed_jaw(fpath):
#       dist_list = []
#       for i in range(0, len(fpath)-1):
#           dist_list.append(
#               dist(fpath[i][0], fpath[i+1][-1], to_km=True)
#           )
#       return(np.mean(dist_list))
#   ct = count_closed_jaw(fpath_trn)
#   print('{},{},{}'.format(car_id, dest_term, ct))

#   # Fit a Gaussian mixture with EM using five components
#   gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(dest_trn)
#   plot_results(dest_trn, gmm.predict(dest_trn), gmm.means_, gmm.covariances_, 0,
#              'Gaussian Mixture')

#   # Fit a Dirichlet process Gaussian mixture using five components
#   dpgmm = mixture.BayesianGaussianMixture(n_components=5,
#                                         covariance_type='full').fit(dest_trn)
#   plot_results(dest_trn, dpgmm.predict(dest_trn), dpgmm.means_, dpgmm.covariances_, 1,
#              'Bayesian Gaussian Mixture with a Dirichlet process prior')
#   plt.savefig('tmp.png')


#   def get_avg_dist_of_paths(fpath_all):
#     dists = []
#     for p in fpath_all:
#       dists.append(dist(p[0], p[-1], to_km=True))
#     return np.mean(dists)
#   dst = get_avg_dist_of_paths(fpath_trn)
#   print('{},{},{}'.format(car_id, dest_term, dst))


#   all_trn = np.concatenate(fpath_trn, axis=0)
#   all_tst = np.concatenate(fpath_tst, axis=0)

#   visualize_dest_density(dest_trn, dest_tst, car_id, dest_term, 
#                          density_segment=10, save_dir=VIZ_DIR)

#   kde_kld = kde_divergence(dest_trn, dest_tst, bandwidth=0.01)
#   kde_kld = kde_divergence(all_trn, all_tst, bandwidth=0.01)
#   print('{},{},{}'.format(car_id, dest_term, kde_kld))

  # # split train set into train/validation sets
  # num_trn = int(len(path_trn) * (1 - validation_size))
  # path_trn, path_val = path_trn[:num_trn], path_trn[num_trn:]
  # meta_trn, meta_val = meta_trn[:num_trn], meta_trn[num_trn:]
  # dest_trn, dest_val = dest_trn[:num_trn], dest_trn[num_trn:]

  # cluster_centers = MeanShift(bandwidth=0.01).fit(dest_trn).cluster_centers_
  # print('{},{},{}'.format(car_id, dest_term, len(cluster_centers)))

#   cluster_centers = None
#   cluster_fname = '{}/cluster/car_{}__dest_{}.png'.format(
#           VIZ_DIR, car_id, dest_term)
#   visualize_cluster(dest_trn, dest_val, dest_tst, cluster_centers, 
#                     fname=cluster_fname, kde_kld_x1e5=int(kde_kld*1e5))

  
  

  
