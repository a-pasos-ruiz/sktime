# -*- coding: utf-8 -*-
from datetime import datetime
from abc import abstractmethod
import numpy as np
from numpy import matlib as mb

from sktime.transformations.base import _PanelToTabularTransformer
import time

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["SetCoverDimensionSelection"]


class SetCoverDimensionSelection(_PanelToTabularTransformer):

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.dimensions_selected = None
        self._is_fitted = False
        self.train_time = 0

    def fit(self, X, y=None):
        start = int(round(time.time() * 1000))
        listed_dimensions = self.get_dimension_order(X, y)
        covered = set()
        cover = []

        self.dimensions_selected = []
        # Greedily add the subsets with the most uncovered points
        while len(covered) < 0.4 * len(y) * 0.9:
            subset = max(listed_dimensions, key=lambda s: len(s.set - covered))
            cover.append(subset)
            self.dimensions_selected.append(subset.dimension)
            covered |= subset

        if self.verbose > 0:
            print("Selected dimensions ", len(self.dimensions_selected), " list: ", self.dimensions_selected, " ",
                  datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                  )
        self.train_time = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        return X[:, self.dimensions_selected, :]

    # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    @staticmethod
    def get_elbow(l):
        n_points = len(l)
        all_coord = np.vstack((range(n_points), [d['accuracy'] for d in l])).T
        np.array([range(n_points), [d['accuracy'] for d in l]])
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

    @abstractmethod
    def get_dimension_order(self, X, y):
        pass
