# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["MatthewMiddlehurst", "victordremov"]
__all__ = ["RocketClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket)
from sktime.transformations.panel.dev import DSRocket


class RocketClassifierDS(BaseClassifier):

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        n_jobs=1,
        random_state=None,
        num_dimensions_selected=0,
        num_dimensions=0
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._pipeline = None

        self.num_dimensions_selected = 0,
        self.num_dimensions = 0

        super(RocketClassifierDS, self).__init__()

    def _fit(self, X, y):
        """Build a pipeline containing the Rocket transformer and RidgeClassifierCV.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        _, n_dims, _ = X.shape

        if self.rocket_transform == "rocket":
            rocket = Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self._threads_to_use
            )
        elif self.rocket_transform == "rocket_i":
            rocket = Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self._threads_to_use,
                kernel_type="independent"
            )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        dsRocket = DSRocket()

        self._pipeline = rocket_pipeline = make_pipeline(
            dsRocket,
            rocket,
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        rocket_pipeline.fit(X, y)
        self.num_dimensions=n_dims
        self.num_dimensions_selected= len(dsRocket.dimensions_selected)
        return self

    def _predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._pipeline.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
