# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["AlejandroPasosRuiz", "MatthewMiddlehurst", "victordremov"]
__all__ = ["ROCKETDS"]

import numpy as np
from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.kernel_based import RocketClassifier


class ROCKETDS(BaseClassifier):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(self, random_state=None,
                 n_jobs=1,
                 time_limit_in_minutes=0,
                 ds_train_time = 0,
                 ds_num_selected_dimensions = 0,
                 ds_num_dimensions = 0,
                 ds_transformer=None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.time_limit_in_minutes = time_limit_in_minutes
        self.ds_train_time = ds_train_time
        self.ds_num_selected_dimensions = ds_num_selected_dimensions
        self.ds_num_dimensions = ds_num_dimensions
        self.ds_transformer = ds_transformer
        self._pipeline = None
        super(ROCKETDS, self).__init__()

    def _fit(self, X, y):
        _, n_dims, _ = X.shape
        self._pipeline = hc_pipeline = make_pipeline(
            self.ds_transformer,
            RocketClassifier(random_state=resample_id, rocket_transform="rocket_d")
        )
        hc_pipeline.fit(X, y)
        self.ds_num_dimensions = n_dims
        self.ds_train_time = self.ds_transformer.train_time
        self.ds_num_selected_dimensions = len(self.ds_transformer.dimensions_selected)
        return self

    def _predict(self, X):
        return self._pipeline.predict(X)


