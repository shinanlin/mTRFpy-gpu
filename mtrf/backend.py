"""
Backend selection utility for mTRF GPU/CPU support.
"""
def detect_backend(preferred='auto'):
    """
    Detects and returns the backend module (numpy or cupy) and backend name.
    Args:
        preferred: 'auto', 'gpu', or 'cpu'
    Returns:
        xp: the module (numpy or cupy)
        backend_name: 'gpu' or 'cpu'
    """
    if preferred == 'cpu':
        import numpy as xp
        return xp, 'cpu'
    try:
        import cupy as xp
        # Check if a GPU is available
        _ = xp.zeros(1)
        return xp, 'gpu'
    except Exception:
        import numpy as xp
        return xp, 'cpu'
