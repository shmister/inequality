from params import ur_b

import numpy as np
import quantecon as qe
import time as tm
from scipy.interpolate import interp1d
import pandas as pd


def p_agg(p_agg_ind):
    p00 = p_agg_ind[0,0] + p_agg_ind[0,1]
    p10 = p_agg_ind[2,0] + p_agg_ind[2,1]
    return np.array([[p00, 1 - p00], [p10, 1 - p10]])


def markov_one_step(current_state, probability, trans_matrix):
    if current_state=='L':
        if probability<= trans_matrix[0,0]:
            return 'L'
        else:
            return 'M'
    elif current_state=='M':
        if probability<= trans_matrix[1,0]:
            return 'L'
        elif (probability > trans_matrix[1,0]) & (probability<= trans_matrix[1,0] + trans_matrix[1,1]):
            return 'M'
        else:
            return 'H'
    else:
        if probability <= trans_matrix[2,1]:
            return 'M'
        else:
            return 'H'



def generate_grid(k_min, k_max, n_points, tau=0):
    if tau!=0:
        x = np.linspace(0, 0.5, n_points)
        y = (x/np.max(x))**tau
        return k_min + (k_max-k_min)*y
    else:
        return np.linspace(k_min, k_max, n_points)


def generate_shocks(trans_mat, N, T):
    np.random.seed(int(round(tm.time())))
    # np.random.seed(0)

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

    return emp_shocks, agg_shocks


def generate_shocks0(trans_mat, N, T):

    prob= trans_mat

    ag_shock = np.zeros((T, 1))
    id_shock = np.zeros((T, N))
    np.random.seed(0)

    # ag_shock = np.zeros((T, 1))
    # Transition probabilities between aggregate states
    prob_ag = np.zeros((2, 2))
    prob_ag[0, 0] = prob[0, 0]+prob[0, 1]
    prob_ag[1, 0] = 1-prob_ag[0, 0] # bad state to good state
    prob_ag[1, 1] = prob[2, 2]+prob[2, 3]
    prob_ag[0, 1] = 1-prob_ag[1, 1]

    P = prob/np.kron(prob_ag, np.ones((2, 2)))
    # generate aggregate shocks
    mc = qe.MarkovChain(prob_ag)
    ag_shock = mc.simulate(ts_length=T, init=0)  # start from bad state
    # generate idiosyncratic shocks for all agents in the first period
    draw = np.random.uniform(size=N)
    id_shock[0, :] = draw>ur_b #set state to good if probability exceeds ur_b

    # generate idiosyncratic shocks for all agents starting in second period
    draw = np.random.uniform(size=(T-1, N))
    for t in range(1, T):
        # Fix idiosyncratic itransition matrix conditional on aggregate state
        transition = P[2*ag_shock[t-1]: 2*ag_shock[t-1]+2, 2*ag_shock[t]: 2*ag_shock[t]+2]
        transition_prob = [transition[int(id_shock[t-1, i]), int(id_shock[t-1, i])] for i in range(N)]
        check = transition_prob>draw[t-1, :] #sign whether to remain in current state
        id_shock[t, :] = id_shock[t-1, :]*check + (1-id_shock[t-1, :])*(1-check)

    return id_shock, ag_shock


def lorenz_points(vals_distribution, weights=None):

    if weights is None:
        weights = np.ones(len(vals_distribution))*1/len(vals_distribution)

    df = pd.DataFrame({'values': vals_distribution, 'weights': weights}).sort_values('values', ascending= True)
    df['temp'] = df['weights']*df['values']

    cum_dist = np.cumsum(df['weights'])/np.sum(df['weights']) # cumulative probability distribution
    cum_data = np.cumsum(df['temp'])/np.sum(df['temp']) # cumulative ownership shares

    lorenz_x = np.linspace(0.0,1.0,100)
    lorenz_y = interp1d(cum_dist,cum_data,bounds_error=False,assume_sorted=True)(lorenz_x)

    return lorenz_x, lorenz_y