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


class point:
    """
    A helper class to use points as references
    """

    def __init__(self, p):
        self.p = p
        self.locked_in_step = None # added by Marvin Ernst (2025)

    def get_point(self):
        return self.p


class Zooming(Algorithm):
    """
    The implementation of the Zooming algorithm
    """

    def __init__(self, nu=1, rho=0.9, domain=None, partition=BinaryPartition):
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

        self.active_points = {}
        self.pulled_times = {}
        self.average_rewards = {}

        self.partition.deepen()
        for child in self.partition.get_layer_node_list(depth=1):
            self.make_active(child)

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

        active_arm = point(node.get_cpoint())
        self.active_points[active_arm] = node
        self.pulled_times[active_arm] = 0
        self.average_rewards[active_arm] = 0

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
        for arm in self.active_points.keys():
            arm_r_t = self.average_rewards[arm] + 2 * np.sqrt(
                8 * self.phase / (2 + self.pulled_times[arm])
            )
            if arm_r_t >= maximum_r_t:
                maximum_r_t = arm_r_t
                self.best_arm = arm

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
        self.average_rewards[self.best_arm] = (
            self.average_rewards[self.best_arm] * self.pulled_times[self.best_arm]
            + reward
        ) / (self.pulled_times[self.best_arm] + 1)
        self.pulled_times[self.best_arm] += 1

        self.time += 1

        if self.time >= self.next_end_time:
            self.phase += 1
            self.next_end_time += 2 ** self.phase

        parent = self.active_points[self.best_arm]
        if (
            np.sqrt(8 * self.phase / (2 + self.pulled_times[self.best_arm]))
            <= self.nu * self.rho ** parent.get_depth()
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
    def __init__(self, f, arms, domain=[[0, 1], [0, 1]], nu=1, rho=0.9):
        """
        Wraps the continuous Zooming algorithm for use in a discrete-arm bandit setting.

        Parameters
        ----------
        f : function
            Ground truth reward function used for evaluation.
        arms : np.ndarray
            Array of shape (K, d) representing K discrete arms in d-dimensional space.
        domain : list of list
            The continuous domain over which Zooming is defined, default is 2D unit square.
        nu : float
            Smoothness parameter for the Zooming algorithm.
        rho : float
            Shrinking factor for spatial refinement.
        """
        self.arms = np.array(arms)
        self.zoom = Zooming(nu=nu, rho=rho, domain=domain)
        self.f = f

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
        idx = np.argmin(np.linalg.norm(self.arms - np.array(x), axis=1))
        self.last_selected_point = x  
        return self.arms[idx]

    def get_selected_index(self):
        return np.argmin(np.linalg.norm(self.arms - np.array(self.last_selected_point), axis=1))

    def update(self, arm_idx, reward):
        self.zoom.receive_reward(self.zoom.time, reward)