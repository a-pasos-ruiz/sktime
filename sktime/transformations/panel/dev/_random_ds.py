# -*- coding: utf-8 -*-
from abc import abstractmethod

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import Rocket

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["RandomDimensionSelection"]

from sktime.transformations.base import _PanelToTabularTransformer
from random import sample
import math
import time


class RandomDimensionSelection(_PanelToTabularTransformer):

    def __init__(self, normalise=True, n_jobs=1, random_state=None):
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        self.random_selection = 0.2
        self.dimensions_selected = None
        self._is_fitted = False
        self.train_time = 0

    def fit(self, X, y=None):
        start = int(round(time.time() * 1000))
        _, n, _ = X.shape
        self.dimensions_selected = sample(range(n), math.ceil(n * self.random_selection))
        self.train_time = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        return X[:, self.dimensions_selected, :]
