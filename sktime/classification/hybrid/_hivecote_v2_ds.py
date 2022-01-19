# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["AlejandroPasosRuiz","MatthewMiddlehurst", "victordremov"]
__all__ = ["HIVECOTEV2DS"]

import numpy as np
from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dev import DSRocket
from sktime.classification.hybrid import HIVECOTEV2


class HIVECOTEV2DS(BaseClassifier):

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        n_jobs=1,
        random_state=None,
        num_dimensions_selected=0,
        num_dimensions=0
    ):

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.num_dimensions_selected = 0,
        self.num_dimensions = 0


        self._pipeline = None


        super(HIVECOTEV2DS, self).__init__()

    def _fit(self, X, y):
        _, n_dims, _ = X.shape


        dsRocket = DSRocket()

        self._pipeline = hc_pipeline = make_pipeline(
            dsRocket,
            HIVECOTEV2(random_state=self.random_state,
                         n_jobs=self.n_jobs,
                       time_limit_in_minutes=1440
                       )
        )
        hc_pipeline.fit(X, y)
        self.num_dimensions=n_dims
        self.num_dimensions_selected= len(dsRocket.dimensions_selected)
        return self

    def _predict(self, X):
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._pipeline.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
