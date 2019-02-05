import numpy as np

wd_folder = '/Users/mitya/Desktop/inequality/codes/gitcode/inequality/'

equal_shares = False

beta_mean = 0.9867
beta_sd = 0.0067

# beta_mean = 0.9864
# beta_sd = 0.0053

betaL, betaM, betaH = beta_mean - beta_sd, beta_mean, beta_mean + beta_sd


#betaL, betaM, betaH = 0.9867 - 0.0067, 0.9867, 0.9867+0.0067
#betaL, betaM, betaH = 0.9858, 0.9894, 0.9930


gammaL, gammaM, gammaH = 1.0, 1.0, 1.0  # utility function parameter
alpha = 0.36  # share of capital in production function
delta = 0.025  # depreciation rate
mu = 0.15  # unemployment benefits as a share of the wage
l_bar = 1/0.9  # time endowment; normalizes labor supply to 1 in bad state


Nagents = 1000  # number of agents for stochastic simulation
# J = 1000  # number of grid points for stochastic simulation
Tperiods = 1500 # number of time periods for stochastic simulation

k_min = -2.4 # min capital
k_max = 1000 # max capital
ngridk = 200 # number of grid points
tau = 7 # fine grid parameter

Tperiods_skip = 100 # skip periods

km_min = 30 # aggregate capital min grid point
km_max = 50 # aggregate capital max grid point
ngridkm = 10 # aggregate capital number of grid points


nstates_id = 2    # number of states for the idiosyncratic shock
nstates_ag = 2    # number of states for the aggregate shock


ur_b = 0.1        # unemployment rate in a bad aggregate state
ur_g = 0.04       # unemployment rate in a good aggregate state


delta_a = 0.01 # productivity difference in bad and good states

prob = np.array(([0.525, 0.35, 0.03125, 0.09375],
                 [0.038889, 0.836111, 0.002083, 0.122917],
                 [0.09375, 0.03125, 0.291667, 0.583333],
                 [0.009115, 0.115885, 0.024306, 0.850694]))

prob_type = np.array(([0.995, 0.005, 0.0],
                      [0.000625, 0.99875, 0.000625],
                      [0.0, 0.005, 0.995]))

types_shares = [0.333, 0.334, 0.333]


dif_B = 10**10 # difference between coefficients B of ALM on succ. iter.
criter_k = 1e-8
criter_B = 1e-8
update_k = 0.5
update_B = 0.3
B_init = np.array((0,1))*np.ones((nstates_ag, 1))