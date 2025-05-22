import numpy as np

class RegretTracker:
    def __init__(self, true_means):
        """
        true_means: array of true expected rewards for each arm (shape: [K])
        """
        self.true_means = np.array(true_means)
        self.mu_star = np.max(true_means)
        self.instantaneous_regrets = []
        self.cumulative_regret = 0.0

    def update(self, selected_arm):
        """
        selected_arm: int, index of the arm pulled at current timestep
        """
        mu_a = self.true_means[selected_arm]
        regret = self.mu_star - mu_a
        self.instantaneous_regrets.append(regret)
        self.cumulative_regret += regret

    def get_cumulative_regret(self):
        return self.cumulative_regret

    def get_instantaneous_regrets(self):
        return np.array(self.instantaneous_regrets)

    def reset(self):
        self.instantaneous_regrets = []
        self.cumulative_regret = 0.0