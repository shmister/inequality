from types_model.envment import *
from types_model.individual import *
from types_model.aggregate import *
from types_model.plotting import *

from types_model.params import *

import numpy as np


def main():
    print("Start Main")

    types_shocks = generate_types_shocks(trans_mat= prob_type, N= Nagents, T= Tperiods+Tperiods_skip)



    # env_params = init_env_params()
    # B_new, env_params_updated = update_environment(env_params, B_init)
    # k_prime, _ = init_kprime_kcross(env_params_updated)
    #
    # k_primeL, k_primeM, k_primeH = k_prime.copy(), k_prime.copy(), k_prime.copy()
    #
    # print(np.sum(env_params_updated['id_shocks']))
    # print(np.sum(env_params_updated['agg_shocks']))
    #
    # diff_B= dif_B
    #
    # while diff_B > criter_B:
    #
    #     k_primeLL_new, cLL = individual_optimization(betaL, gammaL, k_primeL, env_params_updated)
    #     k_primeLM_new, cLM = individual_optimization(betaM, gammaM, k_primeL, env_params_updated)
    #
    #     k_primeML_new, cML = individual_optimization(betaL, gammaL, k_primeM, env_params_updated)
    #     k_primeMM_new, cMM = individual_optimization(betaM, gammaM, k_primeM, env_params_updated)
    #     k_primeMH_new, cMH = individual_optimization(betaH, gammaH, k_primeM, env_params_updated)
    #
    #     k_primeHM_new, cHM = individual_optimization(betaM, gammaM, k_primeH, env_params_updated)
    #     k_primeHH_new, cHH = individual_optimization(betaH, gammaH, k_primeH, env_params_updated)
    #
    #
    #     km_series, k_cross_new = aggregate(k_prime_new, env_params_updated)
    #
    #     B_new, env_params_updated = update_environment(env_params, B_new, k_cross_new=k_cross_new, km_ts=km_series)
    #     print("diffB", env_params_updated['diffB'])
    #
    #     diff_B = env_params_updated['diffB']
    #     k_prime = k_prime_new
    #
    # plot_accuracy(km_series, env_params_updated['agg_shocks'], B_new)
    # plot_policy(k_prime_new, km_series, env_params_updated)
    # plot_lorenz(k_cross_new)


if __name__== "__main__":
    main()
