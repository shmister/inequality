from params import *
import numpy as np
from numpy.random import randn
import statsmodels.api as sm
from scipy.optimize import  brentq, root
from scipy.interpolate import RectBivariateSpline, interpn
np.set_printoptions(precision=4, suppress=True)

def solve_ALM(B, env_params):

    id_shocks, agg_shocks = env_params['id_shocks'], env_params['agg_shocks']
    k, km = env_params['km_grid'], env_params['k_grid']

    k_prime = 0.9*k
    n = ngridk*ngridkm*nstates_ag*nstates_id
    k_prime = k_prime.reshape((len(k_prime), 1, 1, 1))
    k_prime = np.ones((ngridk, ngridkm, nstates_ag, nstates_id))*k_prime
    k_prime = k_prime.reshape(n)
    k_cross = np.repeat(k_ss, N)

    """
    Main loop
    Solve for HH problem given ALM
    Generate time series km_ts given policy function
    Run regression and update ALM
    Iterate until convergence
    """
    iteration = 0
    while dif_B > criter_B:
        # Solve for HH policy functions at a given law of motion
        k_prime, c = individual(k_prime, B)
        # Generate time series and cross section of capital
        km_ts, k_cross_1 = aggregate_st(k_cross, k_prime, id_shock, ag_shock)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = np.log(km_ts[burn_in:(T-1)]).flatten()
        X = pd.DataFrame([np.ones(len(x)), ag_shock[burn_in:(T-1)], x, ag_shock[burn_in:(T-1)]*x]).T
        y = np.log(km_ts[(burn_in+1):]).flatten()
        reg = sm.OLS(y, X).fit()
        B_new = reg.params
        B_mat = np.array((B_new[0], B_new[2], B_new[0]+B_new[1], B_new[2]+B_new[3])).reshape((2, 2))
        dif_B = np.linalg.norm(B_mat-B)
        print(dif_B)

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """
        if dif_B > (criter_B*100):
            k_cross = k_cross_1 #replace cross-sectional capital distribution

        B = B_mat*update_B + B*(1-update_B) #update the vector of ALM coefficients
        iteration += 1
    return B, km_ts, k_cross, k_prime, c, id_shock, ag_shock