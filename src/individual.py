import numpy as np

from params import *
import statsmodels.api as sm
from scipy.optimize import  brentq, root
from scipy.interpolate import RectBivariateSpline, interpn


def individual_optimization(k_prime, beta, gamma, env_params):

    n, P = env_params['n'], env_params['P']
    k, km =  env_params['k_grid'], env_params['km_grid']
    K_prime, wage, tax, irate, replacement, wealth = env_params['K_prime'], env_params['wage'], env_params['tax'], env_params['irate'], env_params['replacement'], env_params['wealth']


    k_prime = 0.9*k
    n = ngridk*ngridkm*nstates_ag*nstates_id
    k_prime = k_prime.reshape((len(k_prime), 1, 1, 1))
    k_prime = np.ones((ngridk, ngridkm, nstates_ag, nstates_id))*k_prime
    k_prime = k_prime.reshape(n)


    dif_k = 1
    while dif_k > criter_k:

        k2_prime = np.zeros((n, nstates_ag, nstates_id))
        c_prime = np.zeros((n, nstates_ag, nstates_id))
        mu_prime = np.zeros((n, nstates_ag, nstates_id))
        expec_comp = np.zeros((n, nstates_ag, nstates_id))

        k_prime_reshape = k_prime.reshape((ngridk, ngridkm, nstates_id, nstates_ag))
        K_prime_reshape = K_prime.reshape((ngridk, ngridkm, nstates_id, nstates_ag))
        for i in range(nstates_ag):
            for j in range(nstates_id):

                k2_prime[:, i, j] = RectBivariateSpline(k, km, k_prime_reshape[:, :, i, j]).ev(k_prime_reshape, K_prime_reshape).reshape(n)
                c_prime[:, i, j] = (irate[:, i]*k_prime + replacement[j]*(wage[:, i])+(1-delta)*k_prime-k2_prime[:, i, j]-tax[:, i, j])
                c_prime[:, i, j] = np.maximum(c_prime[:, i, j], 10**(-10))
                mu_prime[:, i, j] = c_prime[:, i, j]**(-gamma)
                expec_comp[:, i, j] = (mu_prime[:, i, j]*(1-delta + irate[:, i]))*P[:, 2*i+j]

        expec = np.sum(expec_comp, axis=(1,2))
        cn = (beta*expec)**(-1/gamma)
        k_prime_n = np.clip(wealth-cn, k_min, k_max)

        dif_k = np.linalg.norm(k_prime_n-k_prime)
        k_prime = update_k*k_prime_n + (1-update_k)*k_prime  # update k_prime_n

    c = wealth - k_prime

    return k_prime, c