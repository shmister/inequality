from types_model.params import *
from types_model.utils import *

import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import RectBivariateSpline, interp1d
import pandas as pd


def plot_accuracy(km_ts, agg_shocks, B):
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

    k_prime = k_prime.reshape((ngridk, ngridkm, nstates_ag, nstates_id))

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


def plot_lorenz(k_cross, k_crossL, k_crossM, k_crossH):

    scf_df = pd.read_pickle(wd_folder + 'data/scf_data.pkl')
    scf_x, scf_y = lorenz_points(vals_distribution=scf_df['networth'], weights=scf_df['wgt'])

    basic_model_df = pd.read_pickle(wd_folder + 'temp/basic_model_lorenz.pkl')
    basic_x0, basic_y0 = basic_model_df['basic_x0'], basic_model_df['basic_y0']

    x0, y0 = lorenz_points(vals_distribution= k_cross)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(basic_x0, basic_y0, label='Basic Model Distribution')
    ax.plot(x0, y0, label='Model Distribution')
    ax.plot(scf_x, scf_y, label='SCF Data')
    ax.set_xlabel('Population Share')
    ax.set_ylabel('Wealth')
    ax.legend(loc='best')
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(k_cross, label='Cross sectional capital distribution', bins=50)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(k_crossL, label='Cross sectional capital distribution', bins=50)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(k_crossM, label='Cross sectional capital distribution', bins=50)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(k_crossH, label='Cross sectional capital distribution', bins=50)
    plt.show()