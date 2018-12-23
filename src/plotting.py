from params import *

import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import RectBivariateSpline


def accuracy_figure(km_ts, agg_shocks, B):
    T = len(km_ts)
    km_alm = np.zeros((T))
    km_alm[0] = km_ts[0]
    for i in range(T-1):
        km_alm[i+1] = np.exp(B[agg_shocks[i], 0] + B[agg_shocks[i], 1]*np.log(km_alm[i]))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(km_ts, label='Implied by policy rule')
    ax.plot(km_alm, label='Aggregate law of motion')
    ax.set_xlabel('Time')
    ax.set_ylabel('Aggregate capital stock')
    ax.legend(loc='best')
    plt.show()


def plot_policy(k_prime, km_ts, env_params):
    k, km = env_params['k_grid'], env_params['km_grid']

    percentiles = [0.1, 0.25, 0.75, 0.9]
    km_percentiles = np.percentile(km_ts, percentiles)
    km_cycler = cycle(km_percentiles)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for a in range(len(km_percentiles)):
        m, n = np.unravel_index(a, (2, 2))
        for i in range(nstates_ag):
            for j in range(nstates_id):

                x_vals = k[0: 40]
                y_vals = RectBivariateSpline(k, km, k_prime[:, :, i, j]).ev(x_vals, next(km_cycler))

                ax[m, n].plot(x_vals, y_vals, label='Aggregate state = %s, Employment = %s' % (i,j))
                ax[m, n].set_xlabel('Capital accumulation: percentile = %s' % (percentiles[a]))
                ax[m, n].legend(loc='best', fontsize=8)
    plt.show()