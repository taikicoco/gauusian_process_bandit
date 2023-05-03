import numpy as np
import scipy.stats as st

class Bandit:
    def __init__(self, bandit_trial = 100, bandit_sample = 100):
        self.train = []
        self.pred = []
        self.bandit_trial = bandit_trial
        self.bandit_sample = bandit_sample

    def gp_ucb(self, train_func, model) -> list:
        for _ in range(self.bandit_trial):
            train_data = np.sort(st.uniform().rvs(self.bandit_sample))
            pred_sampled = model.predict(train_data)
            train = train_data[np.argmax(pred_sampled.isf(0.05))]
            pred = train_func(train)
            
            model.append(self.train, self.pred)
            self.train.append(train)
            self.pred.append(pred)
        return self.train, self.pred

    def gp_ts(self, train_func, model) -> list:
        for _ in range(self.bandit_trial):
            train_data = np.sort(st.uniform().rvs(self.bandit_sample))
            pred_sampled = model.predict(train_data)
            train = train_data[np.argmax(pred_sampled.isf(0.05))]
            p = pred_sampled.rvs()
            train = train_data[np.argmax(p)]
            pred = train_func(train)
            
            model.append(self.train, self.pred)
            self.train.append(train)
            self.pred.append(pred)
        return self.train, self.pred