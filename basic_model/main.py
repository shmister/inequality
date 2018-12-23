from envment import *
from individual import *
from aggregate import *
from plotting import *
import numpy as np


def main():
    print("Start Main")

    env_params = init_env_params()
    B_new, env_params_updated = update_environment(env_params, B_init)
    k_prime, _ = init_kprime_kcross(env_params_updated)

    print(np.sum(env_params_updated['id_shocks']))
    print(np.sum(env_params_updated['agg_shocks']))

    diff_B= dif_B

    while diff_B > criter_B:

        k_prime_new, c = individual_optimization(beta, gamma, k_prime, env_params_updated)

        km_series, k_cross_new = aggregate(k_prime_new, env_params_updated)

        B_new, env_params_updated = update_environment(env_params, B_new, k_cross_new=k_cross_new, km_ts=km_series)
        print("diffB", env_params_updated['diffB'])

        diff_B = env_params_updated['diffB']
        k_prime = k_prime_new

    plot_accuracy(km_series, env_params_updated['agg_shocks'], B_new)
    plot_policy(k_prime_new, km_series, env_params_updated)
    plot_lorenz(k_cross_new)


if __name__== "__main__":
    main()
