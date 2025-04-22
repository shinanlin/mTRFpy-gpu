from .backend import detect_backend
from mtrf_gpu.matrices import covariance_matrices, regularization_matrix, _check_data, _get_xy
import numpy as np  # fallback for non-heavy ops
import random
import sys


def neg_mse(y, y_pred, xp=None):
    """
    Compute negative mean suqare error (mse) between predicted
    and observed data

    Parameters
    ----------
    y: np.ndarray
        samples-by-features matrix of observed data.
    y_pred: np.ndarray
        samples-by-features matrix of predicted data.
    xp: module or None
        Numpy or cupy module for array ops. If None, use numpy.
    Returns
    -------
    neg_mse: np.ndarray
        Negative mse (-mse) for each feature in y.
    """
    if xp is None:
        import numpy as np
        xp = np
    y = xp.asarray(y)
    y_pred = xp.asarray(y_pred)
    mse = xp.mean((y - y_pred) ** 2, axis=0)
    return -mse


def pearsonr(y, y_pred, xp=None):
    """
    Compute Pearson's correlation coefficient between predicted
    and observed data

    Parameters
    ----------
    y: np.ndarray
        samples-by-features matrix of observed data.
    y_pred: np.ndarray
        samples-by-features matrix of predicted data.
    xp: module or None
        Numpy or cupy module for array ops. If None, use numpy.
    Returns
    -------
    r: np.ndarray
        Pearsons r for each feature in y.
    """
    if xp is None:
        import numpy as np
        xp = np
    y = xp.asarray(y)
    y_pred = xp.asarray(y_pred)
    r = xp.mean((y - xp.mean(y, 0)) * (y_pred - xp.mean(y_pred, 0)), 0) / (
        xp.std(y, 0) * xp.std(y_pred, 0)
    )
    return r

def crossval(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    k=-1,
    seed=None,
    average=True,
    verbose=True,
    backend='auto',
):
    """
    GPU/CPU-agnostic version of crossval.
    """
    from collections.abc import Iterable
    xp, backend_name = detect_backend(backend)
    if isinstance(regularization, Iterable) and not isinstance(regularization, (str, bytes)):
        raise ValueError(
            "Crossval only accepts a single scalar for regularization! "
            "For cross-validation with multiple regularization values use `nested_crossval`!"
        )
    if len(stimulus) < 2:
        raise ValueError("Cross-validation requires at least two trials!")
    trf = model.copy()
    if seed is not None:
        random.seed(seed)
    stimulus, response, _ = _check_data(stimulus, response, min_len=2)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
    cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad, trf.preload)
    metric = _crossval(
        model,
        x,
        y,
        cov_xx,
        cov_xy,
        lags,
        fs,
        regularization,
        k,
        average,
        verbose,
        backend=backend,
    )
    return metric

def _crossval(
    model,
    x,
    y,
    cov_xx,
    cov_xy,
    lags,
    fs,
    regularization,
    k,
    average=True,
    verbose=True,
    seed=None,
    backend='auto',
):
    """
    GPU/CPU-agnostic cross-validation. Uses xp (numpy or cupy) for heavy math.
    """
    xp, backend_name = detect_backend(backend)
    if seed is not None:
        random.seed(seed)

    reg_mat_size = x[0].shape[-1] * len(lags) + 1
    regmat = regularization_matrix(reg_mat_size, model.method)
    regmat = xp.asarray(regmat)
    regmat *= regularization / (1 / fs)
    print('[GPU] _crossval: regmat shape:', regmat.shape)
    print('[GPU] _crossval: cov_xx shape:', cov_xx.shape if cov_xx is not None else None)
    print('[GPU] _crossval: cov_xy shape:', cov_xy.shape if cov_xy is not None else None)

    n_trials = len(x)
    k = int(k)
    # print(f"[DEBUG] n_trials: {n_trials}, k: {k}")
    k = min(k, n_trials) if k > 0 else n_trials
    splits = xp.arange(n_trials)
    if hasattr(xp, "asnumpy"):
        splits = xp.asnumpy(splits)
    random.shuffle(splits)
    splits = np.array_split(splits, k)
    print('[GPU] _crossval: splits:', splits)

    if average is True:
        metric = xp.zeros(k)
    else:
        metric = xp.zeros((k, y[0].shape[-1]))

    for isplit in range(len(splits)):
        # print(f"[DEBUG] isplit: {isplit}")
        # print(f"[DEBUG] splits: {splits}")
        train_splits = [splits[j] for j in range(len(splits)) if j != isplit]
        # print(f"[DEBUG] train_splits (len={len(train_splits)}): {train_splits}")
        if len(train_splits) == 0:
            # print(f"[WARNING] No training splits for isplit={isplit}, skipping this split.")
            continue
        idx_val = splits[isplit]
        idx_train = np.concatenate(train_splits)
        if cov_xx is None:
            x_train = [x[i] for i in idx_train]
            y_train = [y[i] for i in idx_train]
            cov_xx_hat, cov_xy_hat = covariance_matrices(
                x_train, y_train, lags, model.zeropad, preload=False
            )
        else:
            cov_xx_hat = cov_xx[idx_train].mean(axis=0)
            cov_xy_hat = cov_xy[idx_train].mean(axis=0)
        cov_xx_hat = xp.asarray(cov_xx_hat)
        cov_xy_hat = xp.asarray(cov_xy_hat)
        w = xp.matmul(xp.linalg.inv(cov_xx_hat + regmat), cov_xy_hat) / (1 / fs)
        trf = model.copy()
        trf.times, trf.bias, trf.fs = xp.array(lags) / fs, w[0:1], fs
        if trf.bias.ndim == 1:
            trf.bias = xp.expand_dims(trf.bias, 1)
        trf.weights = w[1:].reshape(
            (x[0].shape[-1], len(lags), y[0].shape[-1]), order="F"
        )
        x_test, y_test = [x[i] for i in idx_val], [y[i] for i in idx_val]
        # because we are working with covariance matrices, we have to check direction
        # to pass the right variable as stimulus and response to TRF.predict
        if model.direction == 1:
            print('[GPU] _crossval: Predicting with x_test[0] shape:', x_test[0].shape, 'y_test[0] shape:', y_test[0].shape)
            _, metric_test = trf.predict(x_test, y_test, None, average)
        elif model.direction == -1:
            print('[GPU] _crossval: Predicting with y_test[0] shape:', y_test[0].shape, 'x_test[0] shape:', x_test[0].shape)
            _, metric_test = trf.predict(y_test, x_test, None, average)
        print('[GPU] _crossval: metric_test:', metric_test)
        metric[isplit] = metric_test
    print('[GPU] _crossval: metric mean:', metric.mean(axis=0))
    return metric.mean(axis=0)


def nested_crossval(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    bands=None,
    k=-1,
    average=True,
    seed=None,
    verbose=True,
    backend='auto',
):
    """
    GPU/CPU-agnostic version of nested_crossval.
    """
    from collections.abc import Iterable
    xp, backend_name = detect_backend(backend)
    if len(stimulus) < 3:
        raise ValueError("Nested cross-validation requires at least three trials!")
    k = int(k)
    n_trials = len(stimulus)
    k = min(k, n_trials) if k > 0 else n_trials
    stimulus, response, n_trials = _check_data(stimulus, response)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
    if model.method == "banded":
        coefficients = list(product(regularization, repeat=2))
        regularization = [
            banded_regularization(len(lags), c, bands) for c in coefficients
        ]
    # Preload covariance if needed
    if model.preload:
        cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad)
    else:
        cov_xx, cov_xy = None, None
    splits = xp.arange(n_trials)
    if hasattr(xp, "asnumpy"):
        splits = xp.asnumpy(splits)
    random.shuffle(splits)
    splits = np.array_split(splits, k)
    n_splits = len(splits)
    metric_test = xp.zeros(n_splits)
    best_regularization = []
    for split_i in range(n_splits):
        idx_test = splits[split_i]
        idx_train_val = np.concatenate([splits[j] for j in range(n_splits) if j != split_i])
        if not (isinstance(regularization, Iterable) and not isinstance(regularization, (str, bytes))):
            regularization_split_i = regularization
        else:
            metric = xp.zeros(len(regularization))
            for ir in range(len(regularization)):
                if cov_xx is not None:
                    cov_xx_train = cov_xx[idx_train_val, :, :]
                    cov_xy_train = cov_xy[idx_train_val, :, :]
                else:
                    cov_xx_train, cov_xy_train = None, None
                metric[ir] = _crossval(
                    model.copy(),
                    [x[i] for i in idx_train_val],
                    [y[i] for i in idx_train_val],
                    cov_xx_train,
                    cov_xy_train,
                    lags,
                    fs,
                    regularization[ir],
                    k - 1,
                    average=average,
                    verbose=verbose,
                    backend=backend,
                )
            regularization_split_i = list(regularization)[int(xp.argmax(metric))]
        model._train(
            [x[i] for i in idx_train_val],
            [y[i] for i in idx_train_val],
            fs,
            tmin,
            tmax,
            regularization_split_i,
        )
        _, metric_test[split_i] = model.predict(
            [stimulus[i] for i in idx_test], [response[i] for i in idx_test]
        )
        best_regularization.append(regularization_split_i)
    if hasattr(xp, "asnumpy"):
        metric_test = xp.asnumpy(metric_test)
    return metric_test, best_regularization


def _progressbar(it, prefix="", size=50, out=sys.stdout, verbose=True):
    count = len(it)

    def show(j, verbose):
        x = int(size * j / count)
        if verbose:
            print(
                "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
                end="\r",
                file=out,
                flush=True,
            )

    show(0, verbose)
    for i, item in enumerate(it):
        yield item
        show(i + 1, verbose)
    if verbose:
        print("\n", flush=True, file=out)
        
        
def _check_k(k, n_trials):
    if not n_trials > 1:
        raise ValueError("Cross validation requires multiple trials!")
    if n_trials < k:
        raise ValueError("Number of splits can't be greater than number of trials!")
    if k == -1:  # do leave-one-out cross-validation
        k = n_trials
    return k