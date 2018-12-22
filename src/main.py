from envment import *
from individual import *
from aggregate import *
from tests import *
import pandas as pd
import numpy as np
from numpy.random import randn
import statsmodels.api as sm
from scipy.optimize import  brentq, root
from scipy.interpolate import RectBivariateSpline, interpn
# np.set_printoptions(precision=4, suppress=True)

import sys


def main():
    print("Start Main")

    env_params = init_env_params()
    B_new, env_params_updated = update_environment(env_params, B_init)

    print(np.sum(env_params_updated['id_shocks']))
    print(np.sum(env_params_updated['agg_shocks']))

    for i in range(10):

        k_prime_new, c = individual_optimization(beta, gamma, env_params_updated)
        km_series, k_cross_new = aggregate(k_prime_new, env_params_updated)

        B_new, env_params_updated = update_environment(env_params, B_new, k_prime_new=k_prime_new, k_cross_new=k_cross_new, km_ts=km_series)
        print("diffB", env_params_updated['diffB'])





if __name__== "__main__":
    main()
