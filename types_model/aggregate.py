from types_model.utils import *
from types_model.params import *

from scipy.interpolate import interpn
import numpy as np


def aggregate(k_primeL, k_primeM, k_primeH, env_params):

    id_shocks, agg_shocks, types_shocks = env_params['id_shocks'], env_params['agg_shocks'], env_params['types_shocks']
    k, km = env_params['k_grid'], env_params['km_grid']
    epsilon = env_params['epsilon']
    k_cross = env_params['k_cross']
    km_series = np.zeros((Tperiods+Tperiods_skip,1))

    for t in range(Tperiods+Tperiods_skip):
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


        k_primeL_t4 = interpn(points=(k, km, epsilon, epsilon),
                             values=k_primeL.reshape(ngridk, ngridkm, nstates_ag, nstates_id),
                             xi=interp_points).reshape(ngridk, nstates_id)
        k_primeM_t4 = interpn(points=(k, km, epsilon, epsilon),
                              values=k_primeM.reshape(ngridk, ngridkm, nstates_ag, nstates_id),
                              xi=interp_points).reshape(ngridk, nstates_id)
        k_primeH_t4 = interpn(points=(k, km, epsilon, epsilon),
                              values=k_primeH.reshape(ngridk, ngridkm, nstates_ag, nstates_id),
                              xi=interp_points).reshape(ngridk, nstates_id)
        # 4-dimensional capital function at time t is obtained by fixing known
        # km_series[t] and ag_shock

        current_types_shocks = types_shocks[t, :]
        interp_pointsL = np.vstack((k_cross[current_types_shocks=='L'], id_shocks[t,:][current_types_shocks=='L'])).T
        interp_pointsM = np.vstack((k_cross[current_types_shocks=='M'], id_shocks[t,:][current_types_shocks=='M'])).T
        interp_pointsH = np.vstack((k_cross[current_types_shocks=='H'], id_shocks[t,:][current_types_shocks=='H'])).T


        # interp_pointsL = np.vstack((k_cross[0:1000], id_shocks[t,:][0:1000])).T
        # interp_pointsM = np.vstack((k_cross[1000:9000], id_shocks[t,:][1000:9000])).T
        # interp_pointsH = np.vstack((k_cross[9000:10000], id_shocks[t,:][9000:10000])).T

        #print(interp_pointsL.shape, interp_pointsM.shape, interp_pointsH.shape)

        k_crossL_n = interpn(points=(k, epsilon),
                            values= k_primeL_t4.reshape(ngridk, nstates_id),
                            xi= interp_pointsL)

        k_crossM_n = interpn(points=(k, epsilon),
                             values= k_primeM_t4.reshape(ngridk, nstates_id),
                             xi= interp_pointsM)

        k_crossH_n = interpn(points=(k, epsilon),
                             values= k_primeH_t4.reshape(ngridk, nstates_id),
                             xi= interp_pointsH)

        k_cross_n = np.concatenate([k_crossL_n, k_crossM_n, k_crossH_n])

        # restrict k_cross to be within [k_min, k_max]
        k_cross_n = np.clip(k_cross_n, k_min, k_max)
        k_cross = k_cross_n

    return km_series, k_cross, k_crossL_n, k_crossM_n, k_crossH_n