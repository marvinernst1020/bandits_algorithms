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
    GaussianTS
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
    plot_arm_positions
)