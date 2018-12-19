import numpy as np

from scipy.interpolate import interpn


from utils import *
from params import *


def aggregate_st(k_cross, k_prime, env_params):

    id_shocks, agg_shocks = env_params['id_shocks'], env_params['agg_shocks']
    k, km = env_params['km_grid'], env_params['k_grid']
    epsilon = env_params['epsilon']

    km_series = np.zeros((Tperiods,1))

    for t in range(Tperiods):
        """
        find t-th obs. by computing mean of t-th period cross sectional
        distribution of capital
        """
        km_series[t] = np.mean(k_cross)
        km_series[t] = np.clip(km_series[t], km_min, km_max)

        """
        To find km_series[t+1], we should compute a new cross sectional distribution
        at t+1.
        1) Find k_prime by interpolation for realized km_series[t] and agg. shock
        2) Compute new kcross by interpolation given previous kcross and id.shock
        """
        # Stack sampling points for interpolation as len(k)*len(epsilon) x 4
        # arr stores the coordinates of the sampling points: k rows, 4 columns
        interp_points = np.array(np.meshgrid(k, km_series[t], agg_shocks[t], epsilon))
        interp_points = np.rollaxis(interp_points, 0, 5)
        interp_points = interp_points.reshape((len(k)*len(epsilon), 4))

        k_prime_t4 = interpn(points=(k, km, epsilon, epsilon),
                             values=k_prime.reshape(ngridk, ngridkm, nstates_ag, nstates_id),
                             xi=interp_points).reshape(ngridk, nstates_id)
        # 4-dimensional capital function at time t is obtained by fixing known
        # km_series[t] and ag_shock
        interp_points = np.vstack((k_cross, id_shocks[t,:])).T
        """
        given k_cross and idiosyncratic shocks, compute k_cross_n
        """
        k_cross_n = interpn(points=(k, epsilon),
                            values= k_prime_t4.reshape(ngridk, nstates_id),
                            xi= interp_points)

        # restrict k_cross to be within [k_min, k_max]
        k_cross_n = np.clip(k_cross_n, k_min, k_max)
        k_cross = k_cross_n

    return km_series, k_cross