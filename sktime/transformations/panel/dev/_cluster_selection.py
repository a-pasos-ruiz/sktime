# -*- coding: utf-8 -*-
from _datetime import datetime

from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

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
            num_clusters = len(np.unique(y, return_counts=False))
            clf = KMeans(n_clusters=num_clusters, random_state=0).fit(X_)
            y_ = clf.predict(X_)
            accuracy = silhouette_score(X_, y_)
            dimensions.append({"dimension": i, "accuracy": accuracy})
            if self.verbose > 0:
                print("Dimension ", i, " accuracy: ", accuracy, " ",
                      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                      )
        return dimensions
