# -*- coding: utf-8 -*-
"""Hybrid time series classifiers."""
__all__ = [
    "HIVECOTEV1",
    "HIVECOTEV2",
    "HIVECOTEV2DSROCKET",
    "HIVECOTEV2DS",
    "HIVECOTEV2DSRANDOM"
]

from sktime.classification.hybrid._hivecote_v1 import HIVECOTEV1
from sktime.classification.hybrid._hivecote_v2 import HIVECOTEV2
from sktime.classification.hybrid._hivecote_v2_ds_rocket import HIVECOTEV2DSROCKET
from sktime.classification.hybrid._hivecote_v2_random_ds import HIVECOTEV2DSRANDOM
from sktime.classification.hybrid._hivecote_v2_ds import HIVECOTEV2DS
