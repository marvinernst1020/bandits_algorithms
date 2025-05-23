import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_ground_truth(K=10, d=2, sigma=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(K, d))  
    kernel = RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2, random_state=random_state)
    y = gp.sample_y(X, random_state=random_state).flatten()
    gp.fit(X, y)  
    
    def f(x):
        return float(gp.predict(np.array(x).reshape(1, -1)))

    return X, y, f

def generate_multiple_ground_truths(n_trials, **kwargs):
    return [generate_ground_truth(random_state=seed, **kwargs) for seed in range(n_trials)]