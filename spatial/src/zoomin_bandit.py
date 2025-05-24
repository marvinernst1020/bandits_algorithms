import numpy as np
from .zooming import Zooming
from PyXAB.partition.BinaryPartition import BinaryPartition

class CustomObjective:
    def __init__(self, f):
        self.f = f

    def evaluate(self, x):
        return float(self.f(np.array(x)))

def get_zoomin_algorithm(f, domain=[[0, 1], [0, 1]], nu=1.0, rho=0.9, rounds=None):
    algo = Zooming(nu=nu, rho=rho, domain=domain, partition=BinaryPartition)
    return algo