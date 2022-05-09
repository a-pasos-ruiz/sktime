# -*- coding: utf-8 -*-
from _datetime import datetime

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_score
from sktime.transformations.panel.dev._sc import SetCoverDimensionSelection
from sktime.utils.validation.panel import check_X
from sklearn.model_selection import train_test_split, cross_val_predict

from sktime.transformations.panel.dev._ds import DimensionSelection
from sktime.transformations.panel.rocket import MiniRocketMultivariate

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DSRocket"]


class DSRocket(DimensionSelection):

    def get_dimension_order(self, X, y):
        rocket_pipeline = make_pipeline(
            MiniRocketMultivariate(),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        # rocket_pipeline.fit(X, y)
        # X = check_X(X, coerce_to_numpy=True)
        _, n_dims, _ = X.shape
        dimensions = []

        for i in range(n_dims):
            X_ = X[:, i, :]

            y_ = cross_val_predict(rocket_pipeline, X_, y, cv=3)
            accuracy = silhouette_score(X_, y_)
            dimensions.append({"dimension": i, "accuracy": accuracy})
            if self.verbose > 0:
                print("Dimension ", i, " accuracy: ", accuracy, " ",
                      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                      )
        return dimensions
