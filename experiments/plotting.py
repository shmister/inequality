from experiments.params import *
from experiments.utils import *

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


def plot_policy(k_primeL, k_primeM, k_primeH, km_ts, env_params):
    k, km = env_params['k_grid'], env_params['km_grid']

    k_primeL = k_primeL.reshape((ngridk, ngridkm, nstates_ag, nstates_id))
    k_primeM = k_primeM.reshape((ngridk, ngridkm, nstates_ag, nstates_id))
    k_primeH = k_primeH.reshape((ngridk, ngridkm, nstates_ag, nstates_id))

    for km_val in [0, 9]:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        for i in range(nstates_ag):
            for j in range(nstates_id):

                x_vals = k[0: 80]
                y_vals = RectBivariateSpline(k, km, k_primeH[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = H' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)

                y_vals = RectBivariateSpline(k, km, k_primeM[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = M' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)

                y_vals = RectBivariateSpline(k, km, k_primeL[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = L' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)


        plt.show()

    for km_val in [0, 9]:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        for i in range(nstates_ag):
            for j in range(nstates_id):

                x_vals = k[0: 120]
                y_vals = RectBivariateSpline(k, km, k_primeH[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = H' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)

                y_vals = RectBivariateSpline(k, km, k_primeM[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = M' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)

                y_vals = RectBivariateSpline(k, km, k_primeL[:, :, i, j]).ev(x_vals, km[km_val])
                ax[i, j].plot(x_vals, y_vals, label='Agg = %s, Emp = %s, Type = L' % (i,j))
                ax[i, j].set_xlabel('')
                ax[i, j].legend(loc='best', fontsize=8)


        plt.show()


def plot_lorenz(k_cross, k_crossL, k_crossM, k_crossH):

    scf_df = pd.read_pickle(wd_folder + 'data/scf2016.pkl')
    scf_x, scf_y = lorenz_points(vals_distribution=scf_df['networth'], weights=scf_df['wgt'])
    basic_model_df = pd.read_pickle(wd_folder + 'temp/basic_model_lorenz.pkl')
    basic_x0, basic_y0 = basic_model_df['basic_x0'], basic_model_df['basic_y0']

    hetero0_df = pd.read_pickle('/Users/mitya/Desktop/inequality/codes/gitcode/inequality/output/KS_hetero_eqShF/k_cross.pkl')
    x0, y0 = lorenz_points(vals_distribution= hetero0_df['k_cross'])

    x1, y1 = lorenz_points(vals_distribution= k_cross)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(basic_x0, basic_y0, label='Basic KS')
    ax.plot(x0, y0, label='Hetero KS')
    ax.plot(x1, y1, label='Experiment')
    ax.plot(scf_x, scf_y, label='SCF Data')
    ax.set_xlabel('Population Share')
    ax.set_ylabel('Wealth')
    ax.legend(loc='best')
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(k_crossL, label='L', bins=50)
    ax.hist(k_crossM, label='M', bins=50)
    ax.hist(k_crossH, label='H', bins=50)
    ax.set_xlabel('Wealth')
    ax.set_ylabel('Population')
    ax.legend(loc='best')
    plt.show()