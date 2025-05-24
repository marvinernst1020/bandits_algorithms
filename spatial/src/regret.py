import numpy as np

class RegretTracker:
    def __init__(self, true_means: np.ndarray) -> None:
        """
        Parameters
        ----------
        true_means : np.ndarray
            Array of true expected rewards for each arm (shape: [K])
        """
        self.true_means = np.array(true_means)
        self.mu_star = np.max(true_means)
        self.instantaneous_regrets = []
        self.cumulative_regret = 0.0

    def update(self, selected_arm: int) -> None:
        """
        Update the regret after selecting an arm.

        Parameters
        ----------
        selected_arm : int
            Index of the arm pulled at the current timestep
        """
        mu_a = self.true_means[selected_arm]
        regret = self.mu_star - mu_a
        self.instantaneous_regrets.append(regret)
        self.cumulative_regret += regret

    def get_cumulative_regret(self) -> float:
        """
        Returns
        -------
        float
            Total cumulative regret accumulated so far
        """
        return self.cumulative_regret

    def get_instantaneous_regrets(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of instantaneous regrets at each timestep
        """
        return np.array(self.instantaneous_regrets)

    def reset(self) -> None:
        """
        Reset all stored regret information.
        """
        self.instantaneous_regrets = []
        self.cumulative_regret = 0.0