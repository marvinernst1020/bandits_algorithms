import numpy as np

class BernoulliUCB:
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.successes = np.zeros(K)
        self.squared_sums = np.zeros(K)

    def select_arm(self, t):
        if t < self.K:
            return t
        ucb_values = np.zeros(self.K)
        for a in range(self.K):
            n = self.counts[a]
            mu = self.successes[a] / n
            var_hat = mu * (1 - mu)
            bonus = np.sqrt(
                (np.log(t) / n) *
                min(1/4, var_hat + np.sqrt(2 * np.log(t) / n))
            )
            ucb_values[a] = mu + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.successes[arm] += reward

class BernoulliTS:
    def __init__(self, K):
        self.K = K
        self.successes = np.zeros(K)
        self.failures = np.zeros(K)

    def select_arm(self, t=None):
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

class GaussianUCB:
    def __init__(self, K):
        self.K = K
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.squared_sums = np.zeros(K)

    def select_arm(self, t=None):
        if t is None or t < self.K:
            return t if t is not None else np.random.randint(0, self.K)
        ucb_values = np.zeros(self.K)
        for a in range(self.K):
            n = self.counts[a]
            mu = self.means[a]
            var_hat = (self.squared_sums[a] / n - mu**2) if n > 1 else 0
            bonus = np.sqrt(
                (np.log(t or 1) / (n + 1e-8)) *
                min(1/4, var_hat + np.sqrt(2 * np.log(t or 1) / (n + 1e-8)))
            )
            ucb_values[a] = mu + bonus
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        n = self.counts[arm]
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / (n + 1)
        self.squared_sums[arm] += reward ** 2

class GaussianTS:
    def __init__(self, K, prior_mean=0.0, prior_var=1.0, obs_var=1.0):
        self.K = K
        self.obs_var = obs_var
        self.prior_means = np.full(K, prior_mean)
        self.prior_vars = np.full(K, prior_var)
        self.counts = np.zeros(K)
        self.sum_rewards = np.zeros(K)

    def select_arm(self, t=None):
        samples = np.random.normal(self.prior_means, np.sqrt(self.prior_vars))
        return np.argmax(samples)

    def update(self, arm, reward):
        n = self.counts[arm]
        sum_r = self.sum_rewards[arm]

        post_var = 1 / (1 / self.prior_vars[arm] + 1 / self.obs_var)
        post_mean = post_var * (self.prior_means[arm] / self.prior_vars[arm] + reward / self.obs_var)

        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.prior_means[arm] = post_mean
        self.prior_vars[arm] = post_var