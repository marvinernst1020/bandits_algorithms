import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def set_seed(seed: int = 42):
    np.random.seed(seed)

def plot_rewards(true_rewards, estimated_rewards):
    plt.plot(true_rewards, label="True")
    plt.plot(estimated_rewards, label="Estimated")
    plt.legend()
    plt.show()