import numpy as np
import pandas as pd
import time as time

from params import *
from utils import *
from envment import *
from individual import *
from tests import *

def main():
    print("Start Main")

    env_params = gen_env_params()
    env_params_updated = update_environment(B=B_init, env_params= env_params)

    individual_optimization(env_params_updated['k_grid'], beta, gamma, env_params_updated)

    print(len(env_params_updated), len(env_params))

















if __name__== "__main__":
    main()
