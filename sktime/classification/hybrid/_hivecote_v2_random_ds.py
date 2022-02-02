# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""
from datetime import datetime

__author__ = ["AlejandroPasosRuiz", "MatthewMiddlehurst", "victordremov"]
__all__ = ["HIVECOTEV2DSRANDOM"]

import numpy as np
from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dev import RandomDimensionSelection
from sktime.classification.hybrid import HIVECOTEV2


class HIVECOTEV2DSRANDOM(BaseClassifier):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
            self,
            n_jobs=1,
            random_state=None,
            verbose=0
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.num_dimensions_selected = 0,
        self.num_dimensions = 0

        self._pipeline = None

        super(HIVECOTEV2DSRANDOM, self).__init__()

    def _fit(self, X, y):
        _, n_dims, _ = X.shape

        ds = RandomDimensionSelection()

        if self.verbose > 0:
            print("HC2 random started ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        self._pipeline = hc_pipeline = make_pipeline(
            ds,
            HIVECOTEV2(random_state=self.random_state,
                       n_jobs=self.n_jobs,
                       time_limit_in_minutes=720,
                       verbose=1
                       )
        )
        hc_pipeline.fit(X, y)
        self.num_dimensions = n_dims
        self.num_dimensions_selected = len(ds.dimensions_selected)
        return self

    def _predict(self, X):
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._pipeline.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
