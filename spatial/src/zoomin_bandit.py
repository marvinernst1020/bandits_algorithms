import numpy as np
from .zooming import Zooming
from PyXAB.partition.BinaryPartition import BinaryPartition

class CustomObjective:
    def __init__(self, f):
        self.f = f

    def evaluate(self, x):
        return float(self.f(np.array(x)))

def get_zoomin_algorithm(f, domain, rounds=1000):
    objective = CustomObjective(f)
    algo = Zooming(domain=domain, partition=BinaryPartition)
    return algo, objective