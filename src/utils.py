import numpy as np
import quantecon as qe
import time as tm

from params import ur_b

def p_agg(p_agg_ind):
    p00 = p_agg_ind[0,0] + p_agg_ind[0,1]
    p10 = p_agg_ind[2,0] + p_agg_ind[2,1]
    return np.array([[p00, 1 - p00], [p10, 1 - p10]])


def generate_grid(k_min, k_max, n_points, tau=0):
    if tau!=0:
        x = np.linspace(0, 0.5, n_points)
        y = (x/np.max(x))**tau
        return k_min + (k_max-k_min)*y
    else:
        return np.linspace(k_min, k_max, n_points)


def generate_shocks(trans_mat, N, T, Tskip=0):
    np.random.seed(int(round(tm.time())))
    # np.random.seed(0)
    T += Tskip

    agg_trans_mat = p_agg(trans_mat)
    emp_trans_mat = trans_mat/np.kron(agg_trans_mat, np.ones((2, 2)))

    mc = qe.MarkovChain(agg_trans_mat)
    agg_shocks = mc.simulate(ts_length=T, init=0)
    emp_shocks = np.zeros((T, N))

    draw0 = np.random.uniform(size=N)
    emp_shocks[0, :] = draw0>ur_b

    # generate idiosyncratic shocks for all agents starting in second period
    draws = np.random.uniform(size=(T-1, N))
    for t in range(1, T):
        curr_emp_trans_mat = emp_trans_mat[2*agg_shocks[t-1]: 2*agg_shocks[t-1]+2, 2*agg_shocks[t]: 2*agg_shocks[t]+2]
        curr_emp_trans_probs =  np.where(emp_shocks[t-1, :]==0.0, curr_emp_trans_mat[0,0], curr_emp_trans_mat[1,1])
        emp_shocks[t, :] = np.where(curr_emp_trans_probs>draws[t-1, :], emp_shocks[t-1, :], 1 - emp_shocks[t-1, :])

    return emp_shocks[Tskip:, ], agg_shocks[Tskip:]
