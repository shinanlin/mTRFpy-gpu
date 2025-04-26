import time
from mtrf_gpu.model import TRF
import numpy as np
from mtrf_gpu.stats import crossval, nested_crossval, permutation_distribution

def make_fake_data(n_trials=20, n_samples=128, n_features=30, n_channels=5):
    import numpy as np
    stimulus = [np.random.randn(n_features, n_samples) for _ in range(n_trials)]
    response = [np.random.randn(n_channels, n_samples) for _ in range(n_trials)]
    
    stimulus = np.stack(stimulus, axis=0)
    response = np.stack(response, axis=0)
    
    # stimulus = [stimulus]
    # response = [response]
    
    return stimulus, response


def split_loo_dataset(X, y, n_split=5, random_state=1):
    import numpy as np
    # return leave-one-out dataset
    if n_split == 1:
        return X, y
    # shuffle X,y first
    n_trials, n_channels, n_times = X.shape
    np.random.seed(random_state)
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # split the dataset

    indices = np.arange(n_trials // n_split, n_trials, step=n_trials // n_split).tolist()
    X = X.transpose((1, 0, 2))
    X = np.split(X, indices, axis=1)
    X = [x.reshape((x.shape[0], -1)).T for x in X]
    
    y = y.transpose((1, 0, 2))
    y = np.split(y, indices, axis=1)
    y = [x.reshape((x.shape[0], -1)).T for x in y]
    
    return X, y

def run_test(backend):
    print(f"\nTesting backend: {backend.upper()}")
    print("[INFO] Generating fake data...")
    stimulus, response = make_fake_data()
    print("[INFO] Data generated.")
    fs = 64
    tmin, tmax = -0.2, 0.6
    regularization = 1.0
    
    print('Data size is' , stimulus.shape, response.shape)
    print('FS is', fs)
    print('Tmin is', tmin)
    print('Tmax is', tmax)

    stimulus, response = split_loo_dataset(stimulus, response, n_split=3)
    
    
    print('Stimulus size is', stimulus[0].shape, response[0].shape)
    
    # print("[INFO] Initializing TRF model...")
    model = TRF(backend=backend)
    # print("[INFO] Starting model.train()...")
    # start = time.time()
    # model.train(stimulus, response, fs, tmin, tmax, regularization)
    # print(f"[INFO] Train time: {time.time() - start:.2f} s")

    print("[INFO] Starting cross-validation...")
    # start = time.time()
    # metric = crossval(model, stimulus, response, fs, tmin, tmax, regularization, k=2, backend=backend)
    # print(f"[INFO] Crossval metric: {metric}")
    # print(f"[INFO] Crossval time: {time.time() - start:.2f} s")

    print("[INFO] Starting nested cross-validation...")
    start = time.time()
    n_permute = 6
    result = permutation_distribution(
        model, stimulus, response, fs, tmin, tmax, regularization, n_permute, average=False
    )
    result = np.stack(result)
    print(result.shape)
    # metric_nested, best_reg = nested_crossval(model, stimulus, response, fs, tmin, tmax, [0.1, 1.0, 10.0], k=5, backend=backend)
    # print(f"[INFO] Nested crossval metric: {metric_nested}, best reg: {best_reg}")
    print(f"[INFO] Nested crossval time: {time.time() - start:.2f} s")
    print(f"[INFO] Nested crossval result: {result}")

if __name__ == "__main__":
    run_test('gpu')
    # run_test('cpu')
