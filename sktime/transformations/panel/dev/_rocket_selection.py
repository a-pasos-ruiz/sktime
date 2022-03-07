# -*- coding: utf-8 -*-
from _datetime import datetime

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from sktime.transformations.panel.dev._sc import SetCoverDimensionSelection
from sktime.utils.validation.panel import check_X
from sklearn.model_selection import train_test_split

from sktime.transformations.panel.dev._ds import DimensionSelection
from sktime.transformations.panel.rocket import MiniRocketMultivariate

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DSRocket"]


class DSRocket(SetCoverDimensionSelection):

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y)
        for i in range(n_dims):
            X_train_ = X_train[:, i, :]
            X_test_ = X_test[:, i, :]

            rocket_pipeline.fit(X_train_, y_train)
            accuracy = rocket_pipeline.score(X_test_, y_test)
            y_pred = rocket_pipeline.predict(X_test_)

            dimensions.append({"dimension": i, "accuracy": accuracy,
                               "set": [i for i in range(len(y)) if y_test[i] == y_pred[i]]})
            if self.verbose > 0:
                print("Dimension ", i, " accuracy: ", accuracy, " ",
                      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                      )
        return dimensions
