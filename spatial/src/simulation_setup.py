import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_ground_truth(K=10, d=2, sigma=0.1, random_state=None, length_scale=0.2, scale_factor = 3.0, bias = 0.0):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(K, d))  
    kernel = RBF(length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2, random_state=random_state)
    y = bias + scale_factor * gp.sample_y(X, random_state=random_state).flatten()
    gp.fit(X, y)  
    
    def f(x):
        x = np.atleast_2d(x).astype(np.float64)
        if x.shape[1] != gp.X_train_.shape[1]:
            raise ValueError(f"Input x has {x.shape[1]} features, but GP expects {gp.X_train_.shape[1]}. x: {x}")
        return float(gp.predict(x))

    return X, y, f

def generate_multiple_ground_truths(n_trials, **kwargs):
    return [generate_ground_truth(random_state=seed, **kwargs) for seed in range(n_trials)]