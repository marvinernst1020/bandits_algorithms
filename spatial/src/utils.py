import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.special import expit, logit
import seaborn as sns
import pandas as pd

def set_seed(seed: int = 42):
    np.random.seed(seed)

def plot_rewards(true_rewards, estimated_rewards):
    plt.plot(true_rewards, label="True")
    plt.plot(estimated_rewards, label="Estimated")
    plt.legend()
    plt.show()


def plot_cumulative_regret(df):
    plt.figure(figsize=(8, 5))
    
    # Define consistent colors
    color_mapping = {
        "TS": "blue", "ZI-TS": "blue",
        "UCB0": "green", "ZI-UCB0": "green",
        "UCB": "grey", "ZI-UCB1": "red",
    }

    # Define linestyles
    linestyle_mapping = {
        "TS": "--", "ZI-TS": "-",
        "UCB0": "--", "ZI-UCB0": "-",
        "UCB": "-", "ZI-UCB1": "-",
    }

    # Sort algorithms for consistent legend order
    algorithms = sorted(df["algorithm"].unique())

    # Plot each algorithm individually for full control
    for algo in algorithms:
        algo_df = df[df["algorithm"] == algo]
        sns.lineplot(
            data=algo_df,
            x="time",
            y="avg_regret",
            label=algo,
            color=color_mapping.get(algo, "gray"),
            linestyle=linestyle_mapping.get(algo, "-")
        )

    plt.xlabel("Time Step")
    plt.ylabel("Average Cumulative Regret")
    plt.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(algorithms),
        frameon=False
    )
    plt.tight_layout()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

def plot_cumulative_regret_logit(df, palette=None):
    plt.figure(figsize=(8, 5))

    # Normalize regret to (0, 1)
    df = df.copy()
    max_regret = df["avg_regret"].max()
    epsilon = 1e-5  # avoid logit(0) or logit(1)
    df["regret_scaled"] = df["avg_regret"].clip(epsilon, max_regret - epsilon) / max_regret
    df["regret_logit"] = logit(df["regret_scaled"])

    sns.lineplot(data=df, x="time", y="regret_logit", hue="algorithm", palette=palette)
    plt.xlabel("Time Step")
    plt.ylabel("Logit-Scaled Cumulative Regret")

    plt.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15), 
        ncol=len(df["algorithm"].unique()),  
        frameon=False
    )
    plt.tight_layout()
    plt.show()

def plot_instantaneous_regret(df, palette=None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="time",
        y="avg_inst_regret",
        hue="algorithm",
        palette=palette,
        alpha=0.7  
    )
    plt.xlabel("Time Step")
    plt.ylabel("Average Instantaneous Regret")
    plt.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(df["algorithm"].unique()),
        frameon=False
    )
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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

