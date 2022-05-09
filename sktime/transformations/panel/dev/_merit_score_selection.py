# -*- coding: utf-8 -*-
import math
from _datetime import datetime

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.transformations.panel.dev._sc import SetCoverDimensionSelection
from sktime.utils.validation.panel import check_X
from sklearn.model_selection import train_test_split

from sktime.transformations.panel.dev._ds import DimensionSelection
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import combinations, product, accumulate, chain

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["DSRocket"]


class DSMeritScore(_PanelToTabularTransformer):

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.dimensions_selected = None
        self._is_fitted = False
        self.train_time = 0
        self.predictions = []
        self.feature_to_class = []
        self.feature_to_feature = []

    def fit(self, X, y=None):
        start = int(round(time.time() * 1000))
        _, n_dims, _ = X.shape
        rocket_pipeline = make_pipeline(
            MiniRocketMultivariate(),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.feature_to_feature = a = [[0 for x in range(n_dims)] for y in range(n_dims)]
        for i in range(n_dims):
            X_ = X[:, i, :]

            self.predictions[i] = cross_val_predict(rocket_pipeline, X_, y, cv=3)
            self.feature_to_class[i] = adjusted_mutual_info_score(y, self.predictions[i])

            if self.verbose > 0:
                print("feature ", i, "to-class: ", feature_to_class[i], " ",
                      datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                      )

        for i in range(n_dims):
            for j in (j for j in range(n_dims) if i < j):
                self.feature_to_feature[i][j] = adjusted_mutual_info_score(self.predictions[i], self.predictions[j])
                self.feature_to_feature[j][i] = self.feature_to_feature[i][j]

        self.dimensions_selected = self.forward_selection(n_dims)
        # if self.verbose > 0:
        #    print("Selected dimensions ", len(self.dimensions_selected), " list: ", self.dimensions_selected, " ",
        #          datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        #          )
        self.train_time = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        return X[:, self.dimensions_selected, :]

    def forward_selection(self, n_dims):
        dims = list(combinations([x for x in range(n_dims)], 1))
        comb = list(combinations([x for x in range(n_dims)], 2))

        subsets = [{"subset":subset,"score":self.get_score(subset)} for subset in comb)]
        subsets.sort(key=lambda x: x['score'], reverse=True)
        best_score = subsets[0].score
        best_score_new = subsets[0].score
        id_dim = self.get_elbow(subsets) + 1
        subsets_list = [d['subset'] for d in subsets[:id_dim]]

        while best_score<=best_score_new:
            subsets_list  = [list((map(lambda x: x + y, subsets_list))) for y in dims]
            subsets_list = [item for sublist in subsets_list for item in sublist]
            subsets_list2 = list(filter(lambda x: len(x) == len(set(x)), subsets_list))

            subsets = [{"subset": subset, "score": self.get_score(subset)} for subset in subsets_list2)]
            subsets.sort(key=lambda x: x['score'], reverse=True)
            best_score_new = subsets[0]['score']
            id_dim = self.get_elbow(subsets) + 1
            subsets_list = [d['subset'] for d in subsets[:id_dim]]



    def get_score(self, subset):
        k = len(subset)
        cf = np.mean([self.feature_to_class[i] for i in list(subset)])
        pairs = list(combinations(subset,2))
        ff = np.mean([self.feature_to_feature[t[0]][t[1]] for t in pairs])
        return (k*cf)/(math.sqrt(k+k*(k-1)*ff))

    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    @staticmethod
    def get_elbow(l):
        n_points = len(l)
        all_coord = np.vstack((range(n_points), [d['score'] for d in l])).T
        np.array([range(n_points), [d['score'] for d in l]])
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(vec_from_first * mb.repmat(line_vec_norm, n_points, 1), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        best_point = np.argmax(dist_to_line)
        return best_point