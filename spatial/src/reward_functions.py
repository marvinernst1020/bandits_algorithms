import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

def generate_gp_reward_function(X, sigma=0.1):
    kernel = RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2)
    y = gp.sample_y(X, random_state=0).flatten()
    gp.fit(X, y) 
    return lambda x: float(gp.predict(x.reshape(1, -1))), y