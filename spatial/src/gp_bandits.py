import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GaussianProcessUCB:
    def __init__(self, arms, beta=2.0, noise=0.1):
        self.arms = np.array(arms)
        self.beta = beta
        self.noise = noise
        self.kernel = RBF(length_scale=0.2)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=noise**2)
        self.X = []
        self.y = []

    def select_arm(self, t=None):
        if not self.X:
            return np.random.choice(len(self.arms))
        self.gp.fit(np.array(self.X), np.array(self.y))
        mu, sigma = self.gp.predict(self.arms, return_std=True)
        ucb = mu + self.beta * sigma
        return np.argmax(ucb)

    def update(self, arm_idx, reward):
        self.X.append(self.arms[arm_idx])
        self.y.append(reward)

class GaussianProcessTS:
    def __init__(self, arms, noise=0.1):
        self.arms = np.array(arms)
        self.noise = noise
        self.kernel = RBF(length_scale=0.2)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=noise**2)
        self.X = []
        self.y = []

    def select_arm(self, t=None):
        if not self.X:
            return np.random.choice(len(self.arms))
        self.gp.fit(np.array(self.X), np.array(self.y))
        sampled_f = self.gp.sample_y(self.arms, random_state=None).flatten()
        return np.argmax(sampled_f)

    def update(self, arm_idx, reward):
        self.X.append(self.arms[arm_idx])
        self.y.append(reward)