import numpy as np
import scipy.stats as st

# train data function

class TrainFunc:
    def __init__(self, sigma, f_max, is_func=False):
        self._sigma     = 2 * sigma **2
        self._f_max     = f_max
        self._is_fucn   = is_func

    def peak_one(x):
        func = self._f_max * np.exp(-np.square((x-0.5)/ self._sigma ))
        if self._is_func:
            return func
        return st.bernoulli(func).rvs()
