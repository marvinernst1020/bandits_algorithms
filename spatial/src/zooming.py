# -*- coding: utf-8 -*-
"""Implementation of Zooming (Kleinberg, 2008)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

# linke to the git: https://github.com/WilliamLwj/PyXAB/tree/main/PyXAB/algos

import math
import pdb

import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
from PyXAB.partition.BinaryPartition import BinaryPartition
from scipy.spatial import KDTree



# -----------------------------------
# completely corrected by Marvin Ernst (2025)
class point:
    def __init__(self, p):
        self.p = tuple(np.round(p, 8))  # ensure hash stability

    def get_point(self):
        return np.array(self.p)

    def __hash__(self):
        return hash(self.p)

    def __eq__(self, other):
        return self.p == other.p
# -----------------------------------

class Zooming(Algorithm):
    """
    The implementation of the Zooming algorithm
    """

    def __init__(self, nu=1, rho=0.9, domain=None, partition=BinaryPartition, scoring_method="ucb", reward_type=None, min_pulls_before_zoom=5): # edited by Marvin Ernst (2025)
        """
        Initialization of the Zooming algorithm

        Parameters
        ----------
        nu: float
            smoothness parameter nu of the Zooming algorithm
        rho: float
            smoothness parameter rho of the Zooming algorithm
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm

        Below, added by Marvin Ernst (2025):
        scoring_method: str
            The method for scoring arms ("ucb", "tuned_ucb", "ts")
        reward_type: str
            The type of reward distribution ("gaussian", "bernoulli")
        min_pulls_before_zoom: int
            Minimum number of pulls before zooming in on a node
        """

        super(Zooming, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)

        self.iteration = 1
        self.nu = nu
        self.rho = rho
        self.phase = 1
        self.next_end_time = 2 ** self.phase
        self.time = 0

        self.best_arm = None
        # ----------------------------
        # Added by Marvin Ernst (2025)
        self.scoring_method = scoring_method 
        self.reward_type = reward_type
        self.arms = None  # For discrete-arm search
        self.f = None  # Reward function handle
        # ----------------------------

        self.active_points = {}
        self.pulled_times = {}
        # ----------------------------
        # Edited by Marvin Ernst (2025)
        if self.reward_type == "bernoulli":
            self.successes = {}
            self.failures = {}
        else:
            self.average_rewards = {}

        self.squared_rewards = {}  # initialize the dictionary here
        # ----------------------------

        self.partition.deepen()
        for child in self.partition.get_layer_node_list(depth=1):
            self.make_active(child)

        self.min_pulls_before_zoom = min_pulls_before_zoom

    def make_active(self, node):
        """
        The function to make a node (an arm in the node) active

        Parameters
        ----------
        node:
            the node to be made active

        Returns
        -------
        """
        # ----------------------------
        # Edited by Marvin Ernst (2025)
        # Replacement for discrete-arm search: choose best arm in node's domain, else use center
        if hasattr(self, "arms") and hasattr(self, "tree"):
            # Find best arm within this node's domain
            domain = node.get_domain()
            mask = np.all((self.arms >= [d[0] for d in domain]) & (self.arms <= [d[1] for d in domain]), axis=1)
            candidate_arms = self.arms[mask]
            if len(candidate_arms) > 0:
                best_arm = candidate_arms[np.argmax([self.f(a) for a in candidate_arms])]
                active_arm = point(best_arm)
            else:
                active_arm = point(node.get_cpoint())
        else:
            active_arm = point(node.get_cpoint())
        # ----------------------------
        self.active_points[active_arm] = node
        self.pulled_times[active_arm] = 0
        # ----------------------------
        # Edited by Marvin Ernst (2025)
        if self.reward_type == "bernoulli":
            self.successes[active_arm] = 0
            self.failures[active_arm] = 0
        else:
            self.average_rewards[active_arm] = 0
            self.squared_rewards[active_arm] = 0.0
        # ----------------------------

    def pull(self, time):
        """
        The pull function of Zooming that returns a point in every round

        Parameters
        ----------
        time: int
             time stamp parameter

        Returns
        -------
        point: list
            the point to be evaluated

        """

        maximum_r_t = -np.inf
        self.best_arm = None
        # ----------------------------
        # Edited by Marvin Ernst (2025)
        # Additionaly having the opiton to choose tuned UCB or Thompson Sampling,
        # and not only standard UCB, also havin the option between Bernoulli and Gaussian rewards
        for arm in self.active_points.keys():
            pulls = self.pulled_times[arm]
            if self.reward_type == "bernoulli":
                successes = self.successes[arm]
                failures = self.failures[arm]
                total = successes + failures
                mean = successes / total if total > 0 else 0
            else:
                mean = self.average_rewards[arm]
            if self.scoring_method == "ucb":
                # originally which is should be for the bernoulli rewards, it was:
                # bonus = 2 * np.sqrt(8 * self.phase / (2 + pulls))
                bonus = np.sqrt((2 * np.log(time + 1)) / (pulls + 1e-6))
                score = mean + bonus
            elif self.scoring_method == "tuned_ucb":
                # Gaussian variance estimate
                rewards_squared = self.squared_rewards[arm]
                mean_square = rewards_squared / pulls if pulls > 0 else 0
                variance_estimate = max(0.0, mean_square - mean**2)
                bonus = np.sqrt((np.log(time + 1) / (pulls + 1e-6)) * min(0.25, variance_estimate + np.sqrt((2 * np.log(time + 1)) / (pulls + 1e-6))))
                score = mean + bonus
            elif self.scoring_method == "ts":
                if self.reward_type == "bernoulli":
                    alpha = 1 + self.successes[arm]
                    beta = 1 + self.failures[arm]
                    score = np.random.beta(alpha, beta)
                else:
                    score = np.random.normal(loc=mean, scale=1.0 / np.sqrt(pulls + 1e-6))
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring_method}")

            if score >= maximum_r_t:
                maximum_r_t = score
                self.best_arm = arm
            # ----------------------------

        return self.best_arm.get_point()

    def receive_reward(self, time, reward):
        """
        The receive_reward function of Zooming to obtain the reward and update the statistics, then expand the active arms

        Parameters
        ----------
        time: int
            time stamp parameter
        reward: float
            the reward of the evaluation

        Returns
        -------

        """
        # ----------------------------
        # Added by Marvin Ernst (2025):
        if self.reward_type == "bernoulli":
            if reward > 0.5:
                self.successes[self.best_arm] += 1
            else:
                self.failures[self.best_arm] += 1
        else:
            self.average_rewards[self.best_arm] = (
                self.average_rewards[self.best_arm] * self.pulled_times[self.best_arm]
                + reward
            ) / (self.pulled_times[self.best_arm] + 1)
        self.pulled_times[self.best_arm] += 1
        self.squared_rewards[self.best_arm] += reward**2
        # ----------------------------

        self.time += 1

        if self.time >= self.next_end_time:
            self.phase += 1
            self.next_end_time += 2 ** self.phase
        # ----------------------------
        # Modified by Marvin Ernst (2025)
        parent = self.active_points.get(self.best_arm)
        if parent is None:
            # Try to reconstruct the point key using tuple rounding
            self.best_arm = point(np.round(self.best_arm.get_point(), 8))
            parent = self.active_points[self.best_arm]
        # ----------------------------
        if (
            np.sqrt(8 * self.phase / (2 + self.pulled_times[self.best_arm]))
            <= self.nu * self.rho ** parent.get_depth()
             # ----------------------------
            # Added by Marvin Ernst (2025)
            and self.pulled_times[self.best_arm] >= self.min_pulls_before_zoom
            and all(
                self.pulled_times.get(tuple(child.get_cpoint()), 0) >= self.min_pulls_before_zoom
                for child in (parent.get_children() or [])
            )
            # ----------------------------
        ):
            # ----------------------------
            # Added by Marvin Ernst (2025)
            # Record the first time the algorithm "locks in"
            # Only allow lock-in if at least depth 2 and the arm has been pulled a few times
            if (
                not hasattr(self, "locked_in_step")
                and parent.get_depth() >= 2
                and self.pulled_times[self.best_arm] >= 5
            ):
                self.locked_in_step = self.time
            # ----------------------------
            if parent.get_depth() >= self.partition.get_depth():
                self.partition.make_children(parent=parent, newlayer=True)
            else:
                self.partition.make_children(parent=parent, newlayer=False)

            children_list = parent.get_children()
            for child in children_list:
                child_domain = child.get_domain()
                point = self.best_arm.get_point()

                child_updated = False

                for dim in range(len(child_domain)):
                    if (
                        point[dim] < child_domain[dim][0]
                        or point[dim] > child_domain[dim][1]
                    ):
                        self.make_active(
                            child
                        )  # if not containing the best arm, make the center point active
                        child_updated = True
                        break

                if not child_updated:
                    self.active_points[
                        self.best_arm
                    ] = child  # else, update the active arm to refer to the child node

    def get_last_point(self):
        """
        The function to get the last point of Zooming

        Returns
        -------
        chosen_point: list
            The point chosen by the algorithm
        """

        return self.pull(0)

# -------------------------------------------------------------------------
"""
Added by Marvin Ernst (2025)
"""
# As the Zooming algorithm is not a bandit algorithm, we need to wrap it in a bandit algorithm
# to use it in the bandit setting. The wrapper will select the arm with the minimum distance to the point
# selected by the Zooming algorithm. - for discrete arms:

class DiscreteZoomingWrapper:
    def __init__(self, f, arms, nu, rho, domain, scoring_method="ucb", reward_type=None, min_pulls_before_zoom=5):
        self.f = f
        self.arms = arms
        self.nu = nu
        self.rho = rho
        self.domain = domain
        self.scoring_method = scoring_method
        self.reward_type = reward_type
        self.min_pulls_before_zoom = min_pulls_before_zoom
        self.zoom = Zooming(
            nu=self.nu,
            rho=self.rho,
            domain=self.domain,
            scoring_method=self.scoring_method,
            reward_type=self.reward_type,
            min_pulls_before_zoom=self.min_pulls_before_zoom
        )
        self.zoom.f = self.f
        self.zoom.arms = self.arms
        self.zoom.tree = KDTree(self.arms)
        self.tree = KDTree(self.arms)

    def select_arm(self, t):
        """
        Uses Zooming to propose a point and selects the nearest discrete arm.

        Parameters
        ----------
        t : int
            Current timestep.

        Returns
        -------
        np.ndarray
            The selected arm's coordinates.
        """
        x = self.zoom.pull(t)
        _, idx = self.tree.query(x)
        self.last_selected_point = x
        return self.arms[idx]

    def get_selected_index(self):
        _, idx = self.tree.query(self.last_selected_point)
        return idx

    def update(self, arm_idx, reward):
        self.zoom.receive_reward(self.zoom.time, reward)

    def receive_reward(self, t, reward):
        self.zoom.receive_reward(t, reward)



