import numpy as np

def generate_grid(k_min, k_max, n_points, tau=0):
    if tau!=0:
        x = np.linspace(0, 0.5, n_points)
        y = (x/np.max(x))**tau
        return k_min + (k_max-k_min)*y
    else:
        return np.linspace(k_min, k_max, n_points)