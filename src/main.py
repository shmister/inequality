import numpy as np
import pandas as pd
import time as time

from params import *
from utils import *

from tests import *

def main():
    print("Start Main")

    # generate individual capital grid
    k = generate_grid(k_min, k_max, ngridk, tau)

    # generate aggregate grid
    km = generate_grid(k_min, km_max, ngridkm)


    emp_shocks, agg_shocks = generate_shocks(trans_mat= prob, N= 100, T= 110000, Tskip= t_skip)
    # emp_shocks, agg_shocks = gen_shocks(prob= prob, N= 100, T= 110000)
    print(emp_shocks.shape, agg_shocks.shape)

    test1(agg_shocks, emp_shocks, T)
    print(prob)
















if __name__== "__main__":
    main()
