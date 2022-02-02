# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["AlejandroPasosRuiz", "MatthewMiddlehurst", "victordremov"]
__all__ = ["HIVECOTEV2DSROCKET"]

import numpy as np
from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dev import DSRocket
from sktime.classification.hybrid import HIVECOTEV2


class HIVECOTEV2DSROCKET(BaseClassifier):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(self, random_state=None, n_jobs=1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._pipeline = None
        super(HIVECOTEV2DSROCKET, self).__init__()

    def _fit(self, X, y):
        _, n_dims, _ = X.shape
        ds_rocket = DSRocket(verbose=1)
        self._pipeline = hc_pipeline = make_pipeline(
            ds_rocket,
            HIVECOTEV2(random_state=self.random_state,
                       n_jobs=self.n_jobs,
                       time_limit_in_minutes=720,
                       verbose=1
                       )
        )
        hc_pipeline.fit(X, y)
        self.num_dimensions = n_dims
        self.num_dimensions_selected = len(ds_rocket.dimensions_selected)
        return self

    def _predict(self, X):
        return self._pipeline.predict(X)


