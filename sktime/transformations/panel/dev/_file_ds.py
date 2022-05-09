# -*- coding: utf-8 -*-
from abc import abstractmethod

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import Rocket

__author__ = "Alejandro Pasos Ruiz"
__all__ = ["FileDimensionSelection"]

from sktime.transformations.base import _PanelToTabularTransformer
from random import sample
import math
import time
import csv


class FileDimensionSelection(_PanelToTabularTransformer):

    def __init__(self, results_dir, classifier, dataset, resample):
        self.results_dir = results_dir
        self.classifier = classifier
        self.dataset = dataset
        self.resample = resample
        self.dimensions_selected = None
        self._is_fitted = False
        self.train_time = 0

    def fit(self, X, y=None):
        start = int(round(time.time() * 1000))
        full_path = (
                self.results_dir
                + "/"
                + self.classifier
                + "/ds/"
                + self.dataset
                + "/ds"
                + str(self.resample)
                + ".csv"
        )
        with open('full_path', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.dimensions_selected = [int(i) for i in row]

        self.train_time = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        return X[:, self.dimensions_selected, :]
