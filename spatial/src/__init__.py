# src/__init__.py

# Simulation setup
from .simulation_setup import generate_ground_truth, generate_multiple_ground_truths

# Regret tracking
from .regret import RegretTracker

# Standard bandits
from .standard_bandits import (
    BernoulliUCB,
    BernoulliTS,
    GaussianUCB,
    GaussianTS,
    GaussianUCB0,
    GaussianUCB1
)

# GP bandits
from .gp_bandits import (
    GaussianProcessUCB,
    GaussianProcessTS
)

# Zoom-In
from .zoomin_bandit import get_zoomin_algorithm

# Plotting
from .utils import (
    plot_rewards,
    plot_cumulative_regret,
    plot_cumulative_regret_logit,
    plot_instantaneous_regret,
    ensure_scalar, 
    plot_arm_positions,
    plot_instantaneous_regret_s,
    plot_cumulative_regret_s,
    plot_distance_to_best_arm,
    plot_distance_to_best_arm_s
)