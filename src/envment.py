
import numpy as np

from params import *
from utils import generate_shocks, generate_grid


def gen_env_params():
    # generate individual capital grid
    k = generate_grid(k_min, k_max, ngridk, tau)

    # generate aggregate grid
    km = generate_grid(km_min, km_max, ngridkm)

    # generate idiosyncratic and aggregate shocks
    emp_shocks, agg_shocks = generate_shocks(trans_mat= prob, N= Nagents, T= Tperiods, Tskip= Tperiods_skip)

    a = np.array((1-delta_a, 1+delta_a))
    er_b, er_g = (1-ur_b), (1-ur_g)


    K_ss = ((1/beta-(1-delta))/alpha)**(1/(alpha-1))
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
                  'ag': ag, 'K': K, 'replacement': replacement, 'P': P, 'K_ss': K_ss, 'wealth': wealth}

    return env_params


def update_environment(B, env_params):

    a, e, u, ag, K = env_params['a'], env_params['e'], env_params['u'], env_params['ag'], env_params['K']

    K_prime = np.clip(np.exp(B[ag, 0] + B[ag, 1]*np.log(K)), km_min, km_max)

    irate = alpha*a*((K_prime[:, np.newaxis]/(e.T*l_bar))**(alpha-1))
    wage = (1-alpha)*a*((K_prime[:, np.newaxis]/(e.T*l_bar))**alpha)

    tax_rate = mu*wage*u/(1-u)
    tax = tax_rate[:,:, np.newaxis]*np.array([0,1])

    env_params_updated = env_params.copy()
    env_params_updated.update({'K_prime': K, 'irate': irate, 'wage': wage, 'tax': tax})

    return env_params_updated
