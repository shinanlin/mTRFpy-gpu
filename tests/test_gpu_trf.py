import time
from mtrf.model_gpu import TRF
from mtrf.stats_gpu import crossval, nested_crossval

def make_fake_data(n_trials=20, n_samples=500, n_features=8, n_channels=2):
    import numpy as np
    stimulus = [np.random.randn(n_samples, n_features) for _ in range(n_trials)]
    response = [np.random.randn(n_samples, n_channels) for _ in range(n_trials)]
    return stimulus, response

def run_test(backend):
    print(f"\nTesting backend: {backend.upper()}")
    stimulus, response = make_fake_data()
    fs = 100
    tmin, tmax = -0.1, 0.4
    regularization = 1.0

    model = TRF(backend=backend)
    start = time.time()
    model.train(stimulus, response, fs, tmin, tmax, regularization)
    print(f"Train time: {time.time() - start:.2f} s")

    # Cross-validation
    start = time.time()
    metric = crossval(model, stimulus, response, fs, tmin, tmax, regularization, k=2, backend=backend)
    print(f"Crossval metric: {metric}")
    print(f"Crossval time: {time.time() - start:.2f} s")

    # Nested cross-validation
    start = time.time()
    metric_nested, best_reg = nested_crossval(model, stimulus, response, fs, tmin, tmax, [0.1, 1.0, 10.0], k=5, backend=backend)
    print(f"Nested crossval metric: {metric_nested}, best reg: {best_reg}")
    print(f"Nested crossval time: {time.time() - start:.2f} s")

if __name__ == "__main__":
    run_test('cpu')
    run_test('gpu')
