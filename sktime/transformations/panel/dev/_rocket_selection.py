# -*- coding: utf-8 -*-
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.dev._ds import DimensionSelection
from sktime.transformations.panel.rocket import Rocket

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DSRocket"]

import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X

from numba import njit, get_num_threads, set_num_threads
from numba import prange
from sklearn.model_selection import train_test_split


class DSRocket(DimensionSelection):

    def get_dimension_order(self, X, y):
        rocket_pipeline = make_pipeline(
            Rocket(),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        rocket_pipeline.fit(X, y)
        X = check_X(X, coerce_to_numpy=True)
        _, n_dims, _ = X.shape
        dimensions = []
        for i in range(n_dims):
            X_ = X[:, i, :]
            X_train, X_test, y_train, y_test = train_test_split(
                X_, y, test_size=0.5)
            model = rocket_pipeline.fit(X_train, y_train)
            accuracy = rocket_pipeline.score(X_test, y_test)
            dimensions.append({"dimension": i, "accuracy": accuracy})

        return dimensions
