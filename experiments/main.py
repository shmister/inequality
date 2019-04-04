from experiments.envment import *
from experiments.individual import *
from experiments.aggregate import *
from experiments.plotting import *
from experiments.params import *


def main():

    gammas_list = [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 1.0, 5.0), (0.5, 1.0, 5.0)]
    for gammaL, gammaM, gammaH in gammas_list:

        experiment_folder= experiment_output_dir(gammaL, gammaM, gammaH)

        print("Initiating environment:", print_time())

        env_params = init_env_params(betaL, betaM, betaH)
        B_new, env_params_updated = update_environment(env_params, B_init)
        k_primeL, _ = init_kprime_kcross(env_params_updated)
        k_primeM, _ = init_kprime_kcross(env_params_updated)
        k_primeH, _ = init_kprime_kcross(env_params_updated)

        diff_B= dif_B

        while diff_B > criter_B:

            print("Solving individual optimization:", print_time())
            k_primeL_new, k_primeM_new, k_primeH_new = types_individual_optimzation(k_primeL, k_primeM, k_primeH, env_params_updated, gammaL, gammaM, gammaH)

            print("Solving for aggregates:", print_time())
            km_series, k_cross_new, k_crossL, k_crossM, k_crossH = aggregate(k_primeL_new, k_primeM_new, k_primeH_new, env_params_updated)

            print("Updating environment:", print_time())
            B_new, env_params_updated = update_environment(env_params, B_new, k_cross_new=k_cross_new, km_ts=km_series)
            print("diffB", env_params_updated['diffB'])

            diff_B = env_params_updated['diffB']
            k_primeL, k_primeM, k_primeH = k_primeL_new, k_primeM_new, k_primeH_new

        save_output(k_cross_new,k_crossL, k_crossM, k_crossH, k_primeL_new, k_primeM_new, k_primeH_new, experiment_folder= experiment_folder)


    #
    # plot_accuracy(km_series, env_params_updated['agg_shocks'], B_new)
    # plot_policy(k_primeL_new, k_primeM_new, k_primeH_new, km_series, env_params_updated)
    # plot_lorenz(k_cross_new, k_crossL, k_crossM, k_crossH)


if __name__== "__main__":
    main()
