import numpy as np
import pandas as pd
import time as time

from params import *
from utils import *
from envment import *
from tests import *

def main():
    print("Start Main")

    env_params = gen_env_params()
    update_environment(B=B, k_prime=[0,9], env_params= env_params)

















if __name__== "__main__":
    main()
