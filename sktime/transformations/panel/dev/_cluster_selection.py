# -*- coding: utf-8 -*-
from _datetime import datetime

from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sktime.utils.validation.panel import check_X
from sklearn.model_selection import train_test_split

from sktime.transformations.panel.dev._ds import DimensionSelection
from sktime.transformations.panel.rocket import MiniRocketMultivariate

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DSCluster"]


class DSCluster(DimensionSelection):

    def get_dimension_order(self, X, y):

        _, n_dims, _ = X.shape
        dimensions = []
        for i in range(n_dims):
            X_ = X[:, i, :]

            clf = NearestCentroid(shrink_threshold=0)
            clf.fit(X_, y)
            accuracy = clf.score(X_, y)
            dimensions.append({"dimension": i, "accuracy": accuracy})
            if self.verbose > 0:
                print("Dimension ", i, " accuracy: ", accuracy, " ",
                      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                      )
        return dimensions
