from pathlib import Path
from itertools import product
import pickle
from collections.abc import Iterable
from mtrf.stats import (
    _crossval,
    _progressbar,
    _check_k,
    neg_mse,
    pearsonr,
)
from mtrf.matrices import (
    covariance_matrices,
    banded_regularization,
    regularization_matrix,
    lag_matrix,
    truncate,
    _check_data,
    _get_xy,
)
from .backend import detect_backend

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import mne
except ModuleNotFoundError:
    mne = None

class TRF:
    """
    Temporal response function.
    ... (docstring unchanged) ...
    """
    def __init__(
        self,
        direction=1,
        kind="multi",
        zeropad=True,
        method="ridge",
        preload=True,
        metric=pearsonr,
        backend='auto',  # New: backend selection
    ):
        self.weights = None
        self.bias = None
        self.times = None
        self.fs = None
        self.regularization = None
        if not callable(metric):
            raise ValueError("Metric function must be callable")
        else:
            self.metric = metric
        if isinstance(preload, bool):
            self.preload = preload
        else:
            raise ValueError("Parameter preload must be either True or False!")
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError("Parameter direction must be either 1 or -1!")
        if kind in ["multi", "single"]:
            self.kind = kind
        else:
            raise ValueError('Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = zeropad
        else:
            raise ValueError("Parameter zeropad must be boolean!")
        if method in ["ridge", "tikhonov", "banded"]:
            self.method = method
        else:
            raise ValueError('Method must be either "ridge", "tikhonov" or "banded"!')
        # Backend selection
        self._xp, self.backend = detect_backend(backend)
        print(f"[TRF] Using backend: {self.backend.upper()} ({self._xp.__name__})")


    def copy(self):
        """Return a copy of the TRF instance, preserving backend attributes."""
        # Copy all constructor arguments
        trf = TRF(
            direction=getattr(self, 'direction', 1),
            kind=getattr(self, 'kind', 'multi'),
            zeropad=getattr(self, 'zeropad', True),
            method=getattr(self, 'method', 'ridge'),
            preload=getattr(self, 'preload', True),
            metric=getattr(self, 'metric', None),
            backend=getattr(self, 'backend', 'auto'),
        )
        # Copy all other attributes
        for k, v in self.__dict__.items():
            if k in ['_xp', 'backend']:
                continue  # Already handled by constructor
            value = v
            if getattr(v, "copy", None) is not None:
                value = v.copy()
            setattr(trf, k, value)
        return trf

        self.weights = None
        self.bias = None
        self.times = None
        self.fs = None
        self.regularization = None
        if not callable(metric):
            raise ValueError("Metric function must be callable")
        else:
            self.metric = metric
        if isinstance(preload, bool):
            self.preload = preload
        else:
            raise ValueError("Parameter preload must be either True or False!")
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError("Parameter direction must be either 1 or -1!")
        if kind in ["multi", "single"]:
            self.kind = kind
        else:
            raise ValueError('Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = zeropad
        else:
            raise ValueError("Parameter zeropad must be boolean!")
        if method in ["ridge", "tikhonov", "banded"]:
            self.method = method
        else:
            raise ValueError('Method must be either "ridge", "tikhonov" or "banded"!')
        # Backend selection
        self._xp, self.backend = detect_backend(backend)
        print(f"[TRF] Using backend: {self.backend.upper()} ({self._xp.__name__})")

    @property
    def xp(self):
        return self._xp

    def train(
        self,
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
    ):
        """
        (docstring unchanged)
        """
        xp = self.xp
        if average is False:
            raise ValueError("Average must be True or a list of indices!")
        stimulus, response, n_trials = _check_data(stimulus, response)
        if not xp.isscalar(regularization):
            k = _check_k(k, n_trials)
        x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, self.direction)
        lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
        if self.method == "banded":
            coefficients = list(product(regularization, repeat=len(bands)))
            regularization = [
                banded_regularization(len(lags), c, bands) for c in coefficients
            ]
        if xp.isscalar(regularization):
            self._train(x, y, fs, tmin, tmax, regularization)
            return
        else:  # run cross-validation once per regularization parameter
            # pre-compute covariance matrices
            cov_xx, cov_xy = None, None
            if self.preload:
                cov_xx, cov_xy = covariance_matrices(
                    x, y, lags, self.zeropad, self.preload
                )
            else:
                cov_xx, cov_xy = None, None
            metric = xp.zeros(len(regularization))
            for ir in _progressbar(
                range(len(regularization)),
                "Hyperparameter optimization",
                verbose=verbose,
            ):
                metric[ir] = _crossval(
                    self.copy(),
                    x,
                    y,
                    cov_xx,
                    cov_xy,
                    lags,
                    fs,
                    regularization[ir],
                    k,
                    seed=seed,
                    average=average,
                    verbose=verbose,
                )
            best_regularization = list(regularization)[int(xp.argmax(metric))]
            self._train(x, y, fs, tmin, tmax, best_regularization)
            return metric

    def _train(self, x, y, fs, tmin, tmax, regularization):
        xp = self.xp
        # Ensure regularization is on the right backend
        if isinstance(regularization, xp.ndarray):  # check if matrix is diagonal
            if (
                xp.count_nonzero(regularization - xp.diag(xp.diagonal(regularization)))
                > 0
            ):
                raise ValueError(
                    "Regularization parameter must be a single number or a diagonal matrix!"
                )
        self.fs, self.regularization = fs, regularization
        lags = list(range(int(xp.floor(tmin * fs)), int(xp.ceil(tmax * fs)) + 1))
        cov_xx, cov_xy = covariance_matrices(x, y, lags, self.zeropad, preload=False)
        regmat = regularization_matrix(cov_xx.shape[1], self.method)
        regmat = xp.asarray(regmat)
        regmat *= regularization / (1 / self.fs)
        # Ensure all matrices are on the correct backend
        cov_xx = xp.asarray(cov_xx)
        regmat = xp.asarray(regmat)
        cov_xy = xp.asarray(cov_xy)
        # Main GPU/CPU-agnostic computation
        weight_matrix = xp.matmul(xp.linalg.inv(cov_xx + regmat), cov_xy) / (
            1 / self.fs
        )
        self.bias = weight_matrix[0:1]
        if self.bias.ndim == 1:  # add empty dimension for single feature models
            self.bias = xp.expand_dims(self.bias, axis=0)
        self.weights = weight_matrix[1:].reshape(
            (x[0].shape[1], len(lags), y[0].shape[1]), order="F"
        )
        self.times = xp.array(lags) / fs
        self.fs = fs

    def predict(
        self,
        stimulus=None,
        response=None,
        lag=None,
        average=True,
    ):
        """
        Predict response from stimulus (or vice versa) using the trained model.
        """
        xp = self.xp
        # Faithful to CPU version: input checks, data standardization, and output allocation
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1 and stimulus is None:
            raise ValueError("Need stimulus to predict with a forward model!")
        elif self.direction == -1 and response is None:
            raise ValueError("Need response to predict with a backward model!")
        else:
            stimulus, response, n_trials = _check_data(stimulus, response)
        if stimulus is None:
            stimulus = [None for _ in range(n_trials)]
        if response is None:
            response = [None for _ in range(n_trials)]

        x, y = _get_xy(stimulus, response, direction=self.direction)
        prediction = [xp.zeros((x_i.shape[0], self.weights.shape[-1])) for x_i in x]
        metric = []
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            lags = list(
                range(
                    int(xp.floor(self.times[0] * self.fs)),
                    int(xp.ceil(self.times[-1] * self.fs)) + 1,
                )
            )
            w = self.weights.copy()
            if lag is not None:  # select lag and corresponding weights
                if not isinstance(lag, Iterable):
                    lag = [lag]
                lags = list(xp.array(lags)[lag])
                w = w[:, lag, :]
            w = xp.concatenate(
                [
                    self.bias,
                    w.reshape(
                        x_i.shape[-1] * len(lags), self.weights.shape[-1], order="F"
                    ),
                ]
            ) * (1 / self.fs)
            x_lag = lag_matrix(x_i, lags, self.zeropad, xp=xp)
            # Robustly convert x_lag and w to backend arrays
            if isinstance(x_lag, list):
                x_lag = xp.asarray(xp.stack([xp.asarray(arr) for arr in x_lag]))
            else:
                x_lag = xp.asarray(x_lag)
            if isinstance(w, list):
                w = xp.asarray(xp.stack([xp.asarray(arr) for arr in w]))
            else:
                w = xp.asarray(w)
            y_pred = xp.dot(x_lag, w)
            if y_i is not None:
                if self.zeropad is False:
                    y_i = truncate(y_i, lags[0], lags[-1])
                metric.append(self.metric(y_i, y_pred, xp=self.xp))
            prediction[i][:] = y_pred
        if y[0] is not None:
            metric = xp.mean(xp.stack(metric), axis=0)  # average across trials
            if isinstance(average, list) or (hasattr(xp, 'ndarray') and isinstance(average, xp.ndarray)):
                metric = metric[average]  # select a subset of predictions
            if average is not False:
                metric = xp.mean(metric)
            return prediction, metric
        else:
            return prediction
    # ... other methods unchanged for now ...
