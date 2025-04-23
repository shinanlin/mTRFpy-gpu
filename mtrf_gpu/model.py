from pathlib import Path
from itertools import product
import pickle
from collections.abc import Iterable
from mtrf_gpu.stats import (
    _crossval,
    _progressbar,
    _check_k,
    neg_mse,
    pearsonr,
)
from mtrf_gpu.matrices import (
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
        import time
        time1 = time.time()
        cov_xx, cov_xy = covariance_matrices(x, y, lags, self.zeropad, preload=False, xp=self._xp)
        print(f"[INFO] Covariance computation time: {time.time() - time1:.2f} s")
        regmat = regularization_matrix(cov_xx.shape[1], self.method)
        regmat = xp.asarray(regmat)
        regmat *= regularization / (1 / self.fs)
        # Ensure all matrices are on the correct backend
        cov_xx = xp.asarray(cov_xx)
        regmat = xp.asarray(regmat)
        cov_xy = xp.asarray(cov_xy)
        # Main GPU/CPU-agnostic computation
        import time
        time0 = time.time()
        weight_matrix = xp.matmul(xp.linalg.inv(cov_xx + regmat), cov_xy) / (
            1 / self.fs
        )
        print(f"[INFO] Computation time: {time.time() - time0:.2f} s")
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
        
    def plot(
        self,
        channel=None,
        feature=None,
        axes=None,
        show=True,
        kind="line",
    ):
        """
        Plot the weights of the (forward) model across time for a select channel or feature.

        Arguments:
            channel (None | int | str): Channel selection. If None, all channels will be used. If an integer, the channel at that index will be used. If 'avg' or 'gfp' , the average or standard deviation across channels will be computed.
            feature (None | int | str): Feature selection. If None, all features will be used. If an integer, the feature at that index will be used. If 'avg' , the average across features will be computed.
            axes (matplotlib.axes.Axes): Axis to plot to. If None is provided (default) generate a new plot.
            show (bool): If True (default), show the plot after drawing.
            kind (str): Type of plot to draw. If 'line' (default), average the weights across all stimulus features, if 'image' draw a features-by-times plot where the weights are color-coded.

        Returns:
            fig (matplotlib.figure.Figure): If now axes was provided and a new figure is created, it is returned.
        """
        if plt is None:
            raise ModuleNotFoundError("Need matplotlib to plot TRF!")
        if self.direction == -1:
            weights = self.weights.T
            print(
                "WARNING: decoder weights are hard to interpret, consider using the `to_forward()` method"
            )
        if axes is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = None, axes  # dont create a new figure
        weights = self.weights
        # select channel and or feature
        if weights.shape[0] == 1:
            feature = 0
        if weights.shape[-1] == 1:
            channel = 0
        if channel is None and feature is None:
            raise ValueError("You must specify a subset of channels or features!")
        if feature is not None:
            image_ylabel = "channel"
            if isinstance(feature, int):
                weights = weights[feature, :, :]
            elif feature == "avg":
                weights = weights.mean(axis=0)
            else:
                raise ValueError('Argument `feature` must be an integer or "avg"!')
        if channel is not None:
            image_ylabel = "feature"
            if isinstance(channel, int):
                weights = weights.T[channel].T
            elif channel == "avg":
                weights = weights.mean(axis=-1)
            elif channel == "gfp":
                weights = weights.std(axis=-1)
            else:
                raise ValueError(
                    'Argument `channel` must be an integer, "avg" or "gfp"'
                )
            weights = weights.T  # transpose so first dimension is time
        # plot the result
        if kind == "line":
            ax.plot(
                self.times.flatten(), weights, linewidth=2 - 0.01 * weights.shape[-1]
            )
            ax.set(
                xlabel="Time lag[s]",
                ylabel="Amplitude [a.u.]",
                xlim=(self.times.min(), self.times.max()),
            )
        elif kind == "image":
            scale = self.times.max() / len(self.times)
            im = ax.imshow(
                weights.T,
                origin="lower",
                aspect="auto",
                extent=[0, weights.shape[0], 0, weights.shape[1]],
            )
            extent = np.asarray(im.get_extent(), dtype=float)
            extent[:2] *= scale
            im.set_extent(extent)
            ax.set(
                xlabel="Time lag [s]",
                ylabel=image_ylabel,
                xlim=(self.times.min(), self.times.max()),
            )
        if show is True:
            plt.show()
        if fig is not None:
            return fig

    def to_forward(self, response):
        """
        Transform a backward to a forward model.

        Use the method described in Haufe et al. 2014 to transform the weights of
        a backward model into coefficients reflecting forward activation patterns
        which have a clearer physiological interpretation.

        Parameters
        ----------
        response: list or numpy.ndarray
            response data which was used to train the backward model as single
            trial in a samples-by-channels array or list of multiple trials.

        Returns
        -------
        trf: model.TRF
            New TRF instance with the transformed forward weights
        """
        assert self.direction == -1

        _, response, n_trials = _check_data(None, response)
        stim_pred = self.predict(response=response)

        Cxx = 0
        Css = 0
        trf = self.copy()
        trf.times = np.asarray([-i for i in reversed(trf.times)])
        trf.direction = 1
        for i in range(n_trials):
            Cxx = Cxx + response[i].T @ response[i]
            Css = Css + stim_pred[i].T @ stim_pred[i]
        nStimChan = trf.weights.shape[-1]
        for i in range(nStimChan):
            trf.weights[..., i] = Cxx @ self.weights[..., i] / Css[i, i]

        trf.weights = np.flip(trf.weights.T, axis=1)
        trf.bias = np.zeros(trf.weights.shape[-1])
        return trf

    def to_mne_evoked(self, info, include=None, **kwargs):
        """
        Output TRF weights as instance(s) of MNE-Python's EvokedArray.

        Create one instance of ``mne.EvokedArray`` for each feature along the first
        (i.e. input) dimension of ``self.weights``. When using a backward model,
        the weights are transposed to obtain one EvokedArray per decoded feature.
        See the MNE-Python documentation for details on the Evoked class.

        Parameters
        ----------
        info: mne.Info or mne.channels.montage.DigMontage
            Either a basic info or montage containing channel locations
            Information neccessary to build the EvokedArray.
        include: None or in or list
            Indices of the stimulus features to include. If None (default),
            create one Evoked object for each feature.
        kwargs: dict
            other parameters for constructing the EvokedArray

        Returns
        -------
        evokeds: list
            One Evoked instance for each included TRF feature.
        """
        import numpy as np
        if mne is False:
            raise ModuleNotFoundError("To use this function, mne must be installed!")
        if self.direction == -1:
            weights = self.weights.T
        else:
            weights = self.weights
        if isinstance(info, mne.channels.montage.DigMontage):
            kinds = [d["kind"] for d in info.copy().remove_fiducials().dig]
            ch_types = []
            for k in kinds:
                if "eeg" in str(k).lower():
                    ch_types.append("eeg")
                if "mag" in str(k).lower():
                    ch_types.append("mag")
                if "grad" in str(k).lower():
                    ch_types.append("grad")
            mne_info = mne.create_info(info.ch_names, self.fs, ch_types)
        elif isinstance(info, mne.Info):
            mne_info = info
        else:
            raise ValueError
        if isinstance(include, list) or isinstance(include, np.ndarray):
            weights = weights[np.asarray(include), :, :]
        evokeds = []
        for w in weights:
            evoked = mne.EvokedArray(w.T.copy(), mne_info, tmin=self.times[0], **kwargs)
            if isinstance(info, mne.channels.montage.DigMontage):
                evoked.set_montage(info)
            evokeds.append(evoked)
        return evokeds

    def save(self, path):
        """
        Save class instance using the pickle format.
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"The directory {path.parent} does not exist!")
        with open(path, "wb") as fname:
            pickle.dump(self, fname, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Load pickle file - instance variables will be overwritten with file content.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist!")
        with open(path, "rb") as fname:
            trf = pickle.load(fname)
        self.__dict__ = trf.__dict__


def load_sample_data(path=None, n_segments=1, normalize=True):
    """
    Load sample of brain responses to naturalistic speech.

    If no path is provided, the data is assumed to be in a folder called mtrf_gpu_data
    in the users home directory and will be downloaded and stored there if it can't
    be found. The data contains about 2 minutes of brain responses to naturalistic
    speech, recorded with a 128-channel Biosemi EEG system and the 16-band spectrogram
    of that speech.

    Parameters
    ----------
    path: str or pathlib.Path
        Destination where the sample data is stored or will be downloaded to. If None
        (default), a folder called mtrf_gpu_data in the users home directory is assumed
        and created if it does not exist.

    Returns
    -------
    stimulus: numpy.ndarray
        Samples-by-features array of the presented speech's spectrogram.
    """
    if path is None:  # use default path
        path = Path.home() / "mtrf_gpu_data"
        if not path.exists():
            path.mkdir()
    else:
        path = Path(path)
    if not (path / "speech_data.npy").exists():  # download the data
        url = "https://github.com/powerfulbean/mTRFpy/raw/master/tests/data/speech_data.npy"
        import requests

        response = requests.get(url, allow_redirects=True)
        open(path / "speech_data.npy", "wb").write(response.content)
    data = np.load(str(path / "speech_data.npy"), allow_pickle=True).item()
    stimulus, response, fs = (
        data["stimulus"],
        data["response"],
        data["samplerate"][0][0],
    )
    stimulus = np.array_split(stimulus, n_segments)
    response = np.array_split(response, n_segments)
    if normalize:
        for i in range(len(stimulus)):
            stimulus[i] = (stimulus[i] - stimulus[i].mean(axis=0)) / stimulus[i].std(
                axis=0
            )
            response[i] = (response[i] - response[i].mean(axis=0)) / response[i].std(
                axis=0
            )
    return stimulus, response, fs