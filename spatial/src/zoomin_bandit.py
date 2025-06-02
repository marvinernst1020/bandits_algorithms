import numpy as np
from .zooming import Zooming, DiscreteZoomingWrapper
from PyXAB.partition.BinaryPartition import BinaryPartition

class CustomObjective:
    def __init__(self, f):
        self.f = f  # this is your actual function (e.g., from generate_ground_truth)

    def evaluate(self, x):
        return float(self.f(np.array(x)))  # ensures compatibility with np input

    def __call__(self, x):
        return self.evaluate(x)  # allows object to be used like a function

def get_zoomin_algorithm(f, arms, domain, rounds, nu, rho, scoring_method="ucb", reward_type=None, min_pulls_before_zoom=5):
    return DiscreteZoomingWrapper(
        f=CustomObjective(f),
        arms=arms,
        nu=nu,
        rho=rho,
        domain=domain,
        scoring_method=scoring_method,
        reward_type=reward_type,
        min_pulls_before_zoom=min_pulls_before_zoom
    )

