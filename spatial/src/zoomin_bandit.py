import numpy as np
from .zooming import Zooming
from PyXAB.partition.BinaryPartition import BinaryPartition

class CustomObjective:
    def __init__(self, f):
        self.f = f

    def evaluate(self, x):
        return float(self.f(np.array(x)))

def get_zoomin_algorithm(f, domain, rounds, nu, rho, scoring_method="ucb", reward_type=None):
    return Zooming(nu=nu, rho=rho, domain=domain, scoring_method=scoring_method, reward_type=reward_type)