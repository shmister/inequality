
beta = 0.99  # discount factor
gamma = 1  # utility function parameter
alpha = 0.36  # share of capital in production function
delta = 0.025  # depreciation rate
mu = 0.15  # unemployment benefits as a share of the wage
l_bar = 1/0.9  # time endowment; normalizes labor supply to 1 in bad state


N = 10000  # number of agents for stochastic simulation
J = 1000  # number of grid points for stochastic simulation
k_min = 0 # min capital
k_max = 1000 # max capital
t_skip = 100 # skip periods
T = 1000 + t_skip # simulation time periods
ngridk = 100 # number of grid points

x = np.linspace(0, 0.5, ngridk)
tau = 7
y = (x/np.max(x))**tau
km_min = 30
km_max = 50
k = k_min + (k_max-k_min)*y
ngridkm = 4
km = np.linspace(km_min, km_max, ngridkm)


nstates_id = 2    # number of states for the idiosyncratic shock
nstates_ag = 2    # number of states for the aggregate shock
ur_b = 0.1        # unemployment rate in a bad aggregate state
er_b = (1-ur_b)   # employment rate in a bad aggregate state
ur_g = 0.04       # unemployment rate in a good aggregate state
er_g = (1-ur_g)   # employment rate in a good aggregate state
epsilon = np.arange(0, nstates_id)
delta_a = 0.01
a = np.array((1-delta_a, 1+delta_a))
prob = np.array(([0.525, 0.35, 0.03125, 0.09375],
                 [0.038889, 0.836111, 0.002083, 0.122917],
                 [0.09375, 0.03125, 0.291667, 0.583333],
                 [0.009115, 0.115885, 0.024306, 0.850694]))


dif_B = 10**10 # difference between coefficients B of ALM on succ. iter.
criter_k = 1e-8
criter_B = 1e-8
update_k = 0.77
update_B = 0.3
B = np.array((0,1))*np.ones((nstates_ag, 1))