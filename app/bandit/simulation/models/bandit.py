import numpy as np
import scipy.stats as st

class Bandit:
    def __init__(self, n_play = 100, n_sample = 100):
        self.train = []
        self.reward = []
        self._n_play = n_play
        self._n_sample = n_sample

    def gp_ucb(self, train_func, model):
        for _ in range(self._n_play):
            train_data = np.sort(st.uniform().rvs(self._n_sample))
            reward_sampled = model.predict(train_data)
            train = train_data[np.argmax(reward_sampled.isf(0.05))]
            reward = train_func(train)
            
            model.append(train, reward)
            self.train.append(train)
            self.reward.append(reward)
        return self.train, self.reward

    def gp_ts(self, train_func, model):
        for _ in range(self._n_play):
            train_data = np.sort(st.uniform().rvs(self._n_sample))
            reward_sampled = model.predict(train_data)
            train = train_data[np.argmax(reward_sampled.isf(0.05))]
            p = reward_sampled.rvs()
            train = train_data[np.argmax(p)]
            reward = train_func(train)
            
            model.append(train, reward)
            self.train.append(train)
            self.reward.append(reward)
        return self.train, self.reward