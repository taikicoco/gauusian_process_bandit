import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class RBF:
    def __init__(self, alpha, beta):
        self._alpha = alpha ** 2
        self._beta = 2 * beta ** 2

    def __call__(self, x1, x2):
        if x1.shape == (0,2):
            x1 = np.array([[0,0]])
        d = np.linalg.norm(x1-x2, axis=1, keepdims=True)
        return self._alpha * np.exp(-d / self._beta)

class GP:
    def __init__(self, mu_y, sigma, kernel):
        self._mu_y = mu_y
        self._sigma = sigma ** 2
        self._kernel = kernel
        self._x = np.array([]).reshape(0, 2)
        self._y = np.zeros(0)
        self._K = np.zeros((0, 0))
        self._invK = np.zeros((0, 0))

    def append(self, x, y):
        n = self._y.size
        k0 = self._kernel(x, x)        
        k1 = self._kernel(self._x, x)
        k2 = self._K
        self._K = np.zeros((n + 1, n + 1))
        self._K[:n, :n] = k2
        self._K[n, :n] = k1
        self._K[:n, n] = k1
        self._K[n, n] = k0
        self._x = np.append(self._x, x, axis=0)
        self._y = np.append(self._y, y)
        self._invK = np.linalg.inv(self._K + self._sigma * np.eye(n + 1))

    def dist(self, x):
        k0 = (self._kernel(x, x) + self._sigma)
        k1 = self._kernel(x, self._x).T      
        dy = self._y - self._mu_y
        mean = self._mu_y + k1.T @ self._invK @ dy
        var = k0 - (k1 * (self._invK @ k1)).sum(axis=0)
        return st.norm(mean, np.sqrt(var))
