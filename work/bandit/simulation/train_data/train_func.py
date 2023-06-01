import numpy as np
import scipy.stats as st

# train data function

class TrueFunc:
    def __init__(self, sigma, max_p, is_func=False):
        self._sigma     = 2 * sigma **2
        self._max_p     = max_p
        self._is_func   = is_func

    def peak_one(self, x):
        mean = 0.5
        func = self._max_p * np.exp(-np.square((x - mean)/ self._sigma ))
        if self._is_func:
            return func
        return st.bernoulli(func).rvs()
