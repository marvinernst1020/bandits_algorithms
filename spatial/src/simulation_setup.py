import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def generate_ground_truth(K=10, d=2, sigma=0.1, random_state=None,
                          length_scale=0.2, scale_factor=3.0, bias=0.0,
                          use_custom_f=False, use_bernoulli=False):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(K, d))  # uniform arms

    if use_custom_f:
        margin = 0.1  # how far from 0 or 1
        gap = 0.2     # how far from center (0.5)

        while True:
            center = rng.uniform(0, 1, size=(d,))
            if np.all((center > margin) & (center < 1 - margin)) and np.all(np.abs(center - 0.5) > gap):
                break

        def raw_f(x):
            x = np.atleast_2d(x).astype(np.float64)
            return float(1.0 - np.linalg.norm(x - center))  # Lipschitz

        y_raw = np.array([raw_f(x) for x in X])
        y_min, y_max = y_raw.min(), y_raw.max()
        y = (y_raw - y_min) / (y_max - y_min)

        def f(x):
            val = raw_f(x)
            normalized = (val - y_min) / (y_max - y_min)
            if use_bernoulli:
                return rng.binomial(1, p=np.clip(normalized, 0, 1))
            else:
                return normalized

        return X, y, f
    else:
        kernel = RBF(length_scale)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma**2, random_state=random_state)
        y_raw = bias + scale_factor * gp.sample_y(X, random_state=random_state).flatten()
        y_min, y_max = y_raw.min(), y_raw.max()
        y = (y_raw - y_min) / (y_max - y_min)
        gp.fit(X, y)

        def f(x):
            x = np.atleast_2d(x).astype(np.float64)
            if x.shape[1] != gp.X_train_.shape[1]:
                raise ValueError(f"Input x has {x.shape[1]} features, but GP expects {gp.X_train_.shape[1]}. x: {x}")
            return float(gp.predict(x))

        return X, y, f

def generate_multiple_ground_truths(n_trials, **kwargs):
    return [generate_ground_truth(random_state=seed, **kwargs) for seed in range(n_trials)]


