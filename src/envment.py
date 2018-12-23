from params import *
from utils import generate_shocks, generate_shocks0, generate_grid

import numpy as np
import pandas as pd
import statsmodels.api as sm


def init_env_params():
    # generate individual capital grid
    k = generate_grid(k_min, k_max, ngridk, tau)

    # generate aggregate grid
    km = generate_grid(km_min, km_max, ngridkm)

    # generate idiosyncratic and aggregate shocks
    emp_shocks, agg_shocks = generate_shocks0(trans_mat= prob, N= Nagents, T= Tperiods+Tperiods_skip)

    a = np.array((1-delta_a, 1+delta_a))
    er_b, er_g = (1-ur_b), (1-ur_g)

    k_ss = ((1/beta-(1-delta))/alpha)**(1/(alpha-1))
    P = np.tile(prob, [ngridk*ngridkm, 1])

    e = np.array((er_b, er_g))
    u = 1-e
    replacement = np.array((mu, l_bar)) #replacement rate of wage

    n = ngridk*ngridkm*nstates_ag*nstates_id
    (k_indices, km_indices, ag, e_i) = np.unravel_index(np.arange(n), (ngridk, ngridkm, nstates_ag, nstates_id))
    epsilon = np.arange(0, nstates_id)

    Z, L, K, k_i = a[ag], e[ag], km[km_indices], k[k_indices]

    irate = alpha*Z*(K/(l_bar*L))**(alpha-1)
    wage = (1-alpha)*Z*(K/(l_bar*L))**alpha
    wealth = irate*k_i + (wage*e_i)*l_bar + mu*(wage*(1-e_i))+(1-delta)*k_i-mu*(wage*(1-L)/L)*e_i

    # dictionary
    env_params = {'a': a, 'e': e, 'u': u,
                  'k_grid': k, 'km_grid': km,
                  'n': n, 'epsilon': epsilon,
                  'id_shocks': emp_shocks, 'agg_shocks': agg_shocks,
                  'ag': ag, 'K': K, 'replacement': replacement, 'P': P, 'K_ss': k_ss, 'wealth': wealth, 'B': B_init}

    return env_params


def init_kprime_kcross(env_params):

    n, k, k_ss = env_params['n'], env_params['k_grid'], env_params['K_ss']

    k_prime = 0.9*k
    k_prime = k_prime.reshape((len(k_prime), 1, 1, 1))
    k_prime = np.ones((ngridk, ngridkm, nstates_ag, nstates_id))*k_prime
    k_prime = k_prime.reshape(n)
    k_cross = np.repeat(k_ss, Nagents)

    return k_prime, k_cross


def update_environment(env_params, B_coef, k_cross_new=None, km_ts=None):

    a, e, u, ag, K = env_params['a'], env_params['e'], env_params['u'], env_params['ag'], env_params['K']
    agg_shocks = env_params['agg_shocks']

    if km_ts is None:

        _, k_cross = init_kprime_kcross(env_params)
        diff_B, B_updated = 1000, B_coef

    else:
        x = np.log(km_ts[Tperiods_skip:(Tperiods+Tperiods_skip-1)]).flatten()
        X = pd.DataFrame([np.ones(len(x)), agg_shocks[Tperiods_skip:(Tperiods+Tperiods_skip-1)], x, agg_shocks[Tperiods_skip:(Tperiods+Tperiods_skip-1)]*x]).T
        y = np.log(km_ts[(Tperiods_skip+1):]).flatten()

        reg = sm.OLS(y, X).fit()
        B_new = reg.params
        B_mat = np.array((B_new[0], B_new[2], B_new[0]+B_new[1], B_new[2]+B_new[3])).reshape((2, 2))
        diff_B = np.linalg.norm(B_mat- B_coef)

        k_cross = k_cross_new
        B_updated = B_mat*update_B + B_coef*(1-update_B)


    Kprime = np.clip(np.exp(B_updated[ag, 0] + B_updated[ag, 1]*np.log(K)), km_min, km_max)
    irate = alpha*a*((Kprime[:, np.newaxis]/(e.T*l_bar))**(alpha-1))
    wage = (1-alpha)*a*((Kprime[:, np.newaxis]/(e.T*l_bar))**alpha)

    tax_rate = mu*wage*u/(1-u)
    tax = tax_rate[:,:, np.newaxis]*np.array([0,1])

    env_params_updated = env_params.copy()
    env_params_updated.update({'Kprime': Kprime, 'irate': irate, 'wage': wage, 'tax': tax, 'k_cross': k_cross, 'diffB': diff_B})

    return B_updated, env_params_updated


