import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import seaborn as sns
import pandas as pd

def set_seed(seed: int = 42):
    np.random.seed(seed)

def plot_rewards(true_rewards, estimated_rewards):
    plt.plot(true_rewards, label="True")
    plt.plot(estimated_rewards, label="Estimated")
    plt.legend()
    plt.show()

def plot_cumulative_regret(df, palette=None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="time", y="avg_regret", hue="algorithm", palette=palette)
    plt.xlabel("Time Step")
    plt.ylabel("Average Cumulative Regret")
    plt.legend(title=None, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_instantaneous_regret(df, palette=None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="time", y="avg_inst_regret", hue="algorithm", palette=palette)    
    plt.xlabel("Time Step")
    plt.ylabel("Average Instantaneous Regret")
    plt.legend(title=None, loc="upper right")
    plt.tight_layout()
    plt.show()

def ensure_scalar(x):
    if isinstance(x, (list, np.ndarray)):
        x = np.ravel(x)
        return x[0] if x.size == 1 else np.nan 
    return x

def plot_arm_positions(arms, rewards=None, title="Arm Positions in 2D Space"):
    plt.figure(figsize=(6, 6))
    if rewards is not None:
        sc = plt.scatter(arms[:, 0], arms[:, 1], c=rewards, cmap="viridis", s=100, edgecolors='k')
        plt.colorbar(sc, label="Expected Reward")
    else:
        plt.scatter(arms[:, 0], arms[:, 1], color="blue", s=100, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

