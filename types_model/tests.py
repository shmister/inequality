import numpy as np
import pandas as pd

def test1(agg_shocks, emp_shocks, T):

    agg_shocks_df = pd.DataFrame({
        'tod': agg_shocks[0: T-1],
        'tom': agg_shocks[1:T],
        'shtod': emp_shocks[0: T-1, 3],
        'shtom': emp_shocks[1:T, 3]
    })

    aa = len(agg_shocks_df[(agg_shocks_df['tom']==0) & (agg_shocks_df['tod']==0) & (agg_shocks_df['shtom']==0) & (agg_shocks_df['shtod']==0)])
    ab = len(agg_shocks_df[(agg_shocks_df['tod']==0)  & (agg_shocks_df['shtod']==0)])

    print(aa/ab)