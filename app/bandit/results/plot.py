import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def gp(model, sample, xout, yout, titel = ""):
    xdist = np.sort(st.uniform().rvs(100))
    dist = model.predict(xdist)
    plt.figure(figsize=(5,3))
    plt.plot(xdist, dist.mean(), label='maen')
    plt.plot(xdist, dist.isf(0.05))
    plt.plot(xdist, dist.isf(0.95))
    x = np.linspace(0, 1, 1000)
    plt.plot(x,sample(x), linestyle='--', label='true')
    plt.fill_between(xdist, dist.isf(0.95), dist.isf(0.05), label='95%', color='#0066cc', alpha=0.3)
    plt.scatter(xout, yout, label='data', color='black', marker='o', s=5)
    plt.xlim(0,1)
    plt.title(titel)
    plt.xlabel('select arm')
    plt.ylabel('reward value')
    plt.grid()
    plt.legend()
    plt.show()