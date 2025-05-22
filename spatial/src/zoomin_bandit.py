from PyXAB.algos.Zooming import Zooming
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.synthetic_obj.SyntheticObj import SyntheticObj

class CustomObjective(SyntheticObj):
    def __init__(self, f):
        self.f = f

    def evaluate(self, x):
        return self.f(np.array(x))

def get_zoomin_algorithm(f, domain):
    objective = CustomObjective(f)
    algo = Zooming(rounds=1000, domain=domain, partition=BinaryPartition(domain))
    return algo, objective