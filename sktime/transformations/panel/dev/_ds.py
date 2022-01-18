# -*- coding: utf-8 -*-
from abc import abstractmethod

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import Rocket

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DimensionSelection"]

import multiprocessing

import numpy as np
from numpy import matlib as mb

import pandas as pd

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X

from numba import njit, get_num_threads, set_num_threads
from numba import prange
from sklearn.model_selection import train_test_split


class DimensionSelection(_PanelToTabularTransformer):
    """ Dimension selection


    Parameters
    ----------
    num_kernels  : int, number of random convolutional kernels (default 10,000)
    normalise    : boolean, whether or not to normalise the input time
    series per instance (default True)
    n_jobs             : int, optional (default=1) The number of jobs to run in
    parallel for `transform`. ``-1`` means using all processors.
    random_state : int (ignored unless int due to compatability with Numba),
    random seed (optional, default None)
    """

    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state=None, kernel_type="independent"):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        self.kernel_type = kernel_type
        self.dimensions_selected = None
        self._is_fitted = False

    # super(Rocket, self).__init__()

    def fit(self, X, y=None):
        """Infers time series length and number of channels / dimensions (
        for multivariate time series) from input pandas DataFrame,
        and generates random kernels.

        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)

        Returns
        -------
        self
        """
        listed_dimensions = self.get_dimension_order(X, y)
        listed_dimensions.sort(key=lambda x: x['accuracy'], reverse=True)
        id_dim = self.get_elbow(listed_dimensions)
        if id_dim == 0:
            id_dim = 1
        self.dimensions_selected = [d['dimension'] for d in listed_dimensions[:id_dim]]
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """

        """
        self.check_is_fitted()

        return X[:, self.dimensions_selected, :]

    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    def get_elbow(self, l):
        n_points = len(l)
        all_coord = np.vstack((range(n_points), [d['accuracy'] for d in l])).T
        np.array([range(n_points), [d['accuracy'] for d in l]])
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(vec_from_first * mb.repmat(line_vec_norm, n_points, 1), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        best_point = np.argmax(dist_to_line)
        return best_point

    @abstractmethod
    def get_dimension_order(self, X, y):
        pass
