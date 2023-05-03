import numpy as np
import scipy.stats as st


# 1. インスタンス化するときにalphaとbetaを引数に取る
# 2. インスタンス化したものをカーネルとして扱う
# 3. カーネルは2つの引数を取る
# 4. 2つの引数の差の2乗をbetaで割ったものを負にしたものをalphaで累乗したものを返す
# 5. 3~4を繰り返す
class RBF:
    def __init__(self, alpha, beta):
        self._alpha = alpha ** 2
        self._beta = 2  * beta ** 2
        
    def __call__(self, x1, x2):
        return self._alpha * np.exp(-np.square(x1 - x2) / self._beta)

# 1. インスタンス化するときにmu_yとnoiseとkernelを引数に取る
# 2. インスタンス化したものをガウス過程として扱う
# 3. ガウス過程は2つの引数を取る
# 4. 2つの引数の差の2乗をbetaで割ったものを負にしたものをalphaで累乗したものを返す
# 5. 3~4を繰り返す
class GP:
    def __init__(self, mu_y, noise, kernel):
        self._mu_y = mu_y
        self._noise = noise ** 2
        self._kernel = kernel
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._k = np.zeros((0, 0))
        self._invk = np.zeros((0, 0))
    
    def append(self, x, y):
        n = self._y.size
        k0 = self._kernel(x, x)
        k1 = self._kernel(self._x, x)
        k2 = self._k
        self._k = np.zeros((n + 1, n + 1))
        self._k[:n, :n] = k2
        self._k[n, :n] = k1
        self._k[:n, n] = k1
        self._k[n, n] = k0
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        self._invk = np.linalg.inv(self._k + self._noise * np.eye(n + 1))

    def predict(self, x):
        k0 = self._kernel(x, x) + self._noise
        k1 = self._kernel(x, self._x[:, None])
        dy = self._y - self._mu_y
        mean = self._mu_y + k1.T @ self._invk @ dy
        var = k0 - (k1 * (self._invk @ k1)).sum(axis=0)
        return st.norm(mean, np.sqrt(var))
