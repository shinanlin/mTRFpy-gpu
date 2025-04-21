import numpy as np
import sys
import time

sys.path.append("../mtrf")

from mtrf.model import TRF as TRF_CPU_ORIG
from mtrf.model_gpu import TRF as TRF_GPU

def make_fake_data(seed=42, n_trials=20, n_samples=500, n_features=8, n_channels=2):
    np.random.seed(seed)
    stimulus = [np.random.randn(n_samples, n_features) for _ in range(n_trials)]
    response = [np.random.randn(n_samples, n_channels) for _ in range(n_trials)]
    return stimulus, response

def to_numpy(array):
    """Convert array to numpy, handling CuPy arrays properly"""
    # Check if it's a CuPy array by looking for the get() method
    if hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)

def compare_predictions(y1, y2, name1, name2, atol=1e-6):
    """Compare two lists of predictions and print max absolute diff."""
    for i, (a, b) in enumerate(zip(y1, y2)):
        # Ensure both are numpy arrays
        a_np = to_numpy(a)
        b_np = to_numpy(b)
        diff = np.abs(a_np - b_np)
        max_diff = np.max(diff)
        print(f"  Trial {i}: max abs diff between {name1} and {name2}: {max_diff:.3e}")
        if not np.allclose(a_np, b_np, atol=atol):
            print(f"    WARNING: Predictions differ beyond atol={atol}")

def run_all():
    print("Generating data...")
    stimulus, response = make_fake_data()
    fs = 100
    tmin, tmax = -0.1, 0.4
    regularization = 1.0

    # Instantiate models
    model_cpu_orig = TRF_CPU_ORIG()
    model_cpu_gpu = TRF_GPU(backend='cpu')
    model_gpu = TRF_GPU(backend='gpu')

    print("\nTraining models...")
    for name, model in zip(["CPU_ORIG", "CPU_GPU", "GPU"], [model_cpu_orig, model_cpu_gpu, model_gpu]):
        start = time.time()
        model.train(stimulus, response, fs, tmin, tmax, regularization)
        print(f"  {name} train time: {time.time() - start:.2f} s")

    print("\nPredicting...")
    preds = {}
    metrics = {}
    for name, model in zip(["CPU_ORIG", "CPU_GPU", "GPU"], [model_cpu_orig, model_cpu_gpu, model_gpu]):
        y_pred, metric = model.predict(stimulus, response)
        
        # Handle CuPy arrays properly
        if name == "GPU":
            # For GPU model, convert predictions to NumPy
            preds[name] = [to_numpy(y) for y in y_pred]
            metrics[name] = to_numpy(metric)
        else:
            # For CPU models, just store as is
            preds[name] = [np.asarray(y) for y in y_pred]
            metrics[name] = metric
            
        print(f"  {name} metric: {metric}")

    print("\nComparing predictions and metrics:")
    compare_predictions(preds["CPU_ORIG"], preds["CPU_GPU"], "CPU_ORIG", "CPU_GPU")
    compare_predictions(preds["CPU_ORIG"], preds["GPU"], "CPU_ORIG", "GPU")
    compare_predictions(preds["CPU_GPU"], preds["GPU"], "CPU_GPU", "GPU")

    print("\nMetric differences:")
    print(f"  CPU_ORIG vs CPU_GPU: {np.abs(to_numpy(metrics['CPU_ORIG']) - to_numpy(metrics['CPU_GPU']))}")
    print(f"  CPU_ORIG vs GPU:     {np.abs(to_numpy(metrics['CPU_ORIG']) - to_numpy(metrics['GPU']))}")
    print(f"  CPU_GPU vs GPU:      {np.abs(to_numpy(metrics['CPU_GPU']) - to_numpy(metrics['GPU']))}")

    # Print and compare model weights
    print("\nModel Weights:")
    weights = {}
    for name, model in zip(["CPU_ORIG", "CPU_GPU", "GPU"], [model_cpu_orig, model_cpu_gpu, model_gpu]):
        w = to_numpy(model.weights)
        weights[name] = w
        print(f"{name} weights shape: {w.shape}")
        print(f"{name} weights (first 20 elements): {w.ravel()[:20]}")

    print("\nMax absolute differences between weights:")
    print(f"  CPU_ORIG vs CPU_GPU: {np.max(np.abs(weights['CPU_ORIG'] - weights['CPU_GPU']))}")
    print(f"  CPU_ORIG vs GPU:     {np.max(np.abs(weights['CPU_ORIG'] - weights['GPU']))}")
    print(f"  CPU_GPU vs GPU:      {np.max(np.abs(weights['CPU_GPU'] - weights['GPU']))}")

if __name__ == "__main__":
    run_all()
