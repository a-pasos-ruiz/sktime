# -*- coding: utf-8 -*-
"""Set classifier function."""
from sktime.classification.kernel_based._rocket_classifier_ds import RocketClassifierDS

__author__ = ["TonyBagnall"]

from sklearn.ensemble import RandomForestClassifier

from sktime.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
    ProximityForest,
    ProximityStump,
    ProximityTree,
    ShapeDTW,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCE,
    MatrixProfileClassifier,
    RandomIntervalClassifier,
    SignatureClassifier,
    SummaryClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2, HIVECOTEV2DSRANDOM, \
    HIVECOTEV2DSROCKET, HIVECOTEV2DS, ROCKETDS
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.series.summarize import SummaryTransformer

from sktime.transformations.panel.dev import DSRocket, RandomDimensionSelection, ecs, ecp, DSMeritScore, kmeans, DSCluster, \
    FileDimensionSelection



def set_classifier(cls, resample_id=None, train_file=False,  ,results_dir=None,classifier=None,dataset=None,
                   resample=None):
    """Construct a classifier, possibly seeded.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.

    Parameters
    ----------
    cls : str
        String indicating which classifier you want.
    resample_id : int or None, default=None
        Classifier random seed.
    train_file : bool, default=False
        Whether a train file is being produced.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    name = cls.lower()
    # Dictionary based
    if name == "boss" or name == "bossensemble":
        return BOSSEnsemble(random_state=resample_id)
    elif name == "cboss" or name == "contractableboss":
        return ContractableBOSS(random_state=resample_id)
    elif name == "tde" or name == "temporaldictionaryensemble":
        return TemporalDictionaryEnsemble(
            random_state=resample_id, save_train_predictions=train_file
        )
    elif name == "weasel":
        return WEASEL(random_state=resample_id)
    elif name == "muse":
        return MUSE(random_state=resample_id)
    # Distance based
    elif name == "pf" or name == "proximityforest":
        return ProximityForest(random_state=resample_id)
    elif name == "pt" or name == "proximitytree":
        return ProximityTree(random_state=resample_id)
    elif name == "ps" or name == "proximityStump":
        return ProximityStump(random_state=resample_id)
    elif name == "dtwcv" or name == "kneighborstimeseriesclassifier":
        return KNeighborsTimeSeriesClassifier(distance="dtwcv")
    elif name == "dtw" or name == "1nn-dtw":
        return KNeighborsTimeSeriesClassifier(distance="dtw")
    elif name == "msm" or name == "1nn-msm":
        return KNeighborsTimeSeriesClassifier(distance="msm")
    elif name == "ee" or name == "elasticensemble":
        return ElasticEnsemble(random_state=resample_id)
    elif name == "shapedtw":
        return ShapeDTW()
    # Feature based
    elif name == "summary":
        return SummaryClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "summary-intervals":
        return RandomIntervalClassifier(
            random_state=resample_id,
            interval_transformers=SummaryTransformer(
                summary_function=("mean", "std", "min", "max"),
                quantiles=(0.25, 0.5, 0.75),
            ),
            estimator=RandomForestClassifier(n_estimators=500),
        )
    elif name == "summary-catch22":
        return RandomIntervalClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "catch22":
        return Catch22Classifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "matrixprofile":
        return MatrixProfileClassifier(random_state=resample_id)
    elif name == "signature":
        return SignatureClassifier(
            random_state=resample_id,
            estimator=RandomForestClassifier(n_estimators=500),
        )
    elif name == "tsfresh":
        return TSFreshClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "tsfresh-r":
        return TSFreshClassifier(
            random_state=resample_id,
            estimator=RandomForestClassifier(n_estimators=500),
            relevant_feature_extractor=True,
        )
    elif name == "freshprince":
        return FreshPRINCE(random_state=resample_id, save_transformed_data=train_file)
    # Hybrid
    elif name == "hc1" or name == "hivecotev1":
        return HIVECOTEV1(random_state=resample_id)
    elif name == "hc2" or name == "hivecotev2":
        return HIVECOTEV2(random_state=resample_id, time_limit_in_minutes=0, verbose=1)
    elif name == "hc2-ds-rocket" or name == "hivecotev2dsrocket":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=DSRocket(verbose=1))
    elif name == "hc2-ds-cluster" or name == "hivecotev2dscluster":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=DSCluster(verbose=1))
    elif name == "hc2-ds-ecs" or name == "hivecotev2dsecs":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=ecs())
    elif name == "hc2-ds-kmeans" or name == "hivecotev2dskmeans":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=kmeans())
    elif name == "hc2-ds-ecp" or name == "hivecotev2dsecp":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=ecp())
    elif name == "hc2-ds-merit" or name == "hivecotev2dsmerit":
        return HIVECOTEV2DS(random_state=resample_id, time_limit_in_minutes=0, ds_transformer=DSMeritScore())
    elif name == "hc2-ds-random" or name == "hivecotev2dsrandom":
        return HIVECOTEV2DS(random_state=resample_id,  ds_transformer=RandomDimensionSelection())
    elif name == "hc2-ds-file" or name == "hivecotev2dsfile":
        return HIVECOTEV2DS(random_state=resample_id,
                            ds_transformer=FileDimensionSelection(results_dir, classifier,dataset, resample))
    elif name == "rocket-ds-rocket" or name == "rocketdsrocket":
        return ROCKETDS(random_state=resample_id, ds_transformer=DSRocket())
    # Interval based
    elif name == "rise" or name == "randomintervalspectralforest":
        return RandomIntervalSpectralForest(random_state=resample_id, n_estimators=500)
    elif name == "tsf" or name == "timeseriesforestclassifier":
        return TimeSeriesForestClassifier(random_state=resample_id, n_estimators=500)
    elif name == "cif" or name == "canonicalintervalforest":
        return CanonicalIntervalForest(random_state=resample_id, n_estimators=500)
    elif name == "stsf" or name == "supervisedtimeseriesforest":
        return SupervisedTimeSeriesForest(random_state=resample_id, n_estimators=500)
    elif name == "drcif":
        return DrCIF(
            random_state=resample_id, n_estimators=500, save_transformed_data=train_file
        )
    # Kernel based
    elif name == "rocket":
        return RocketClassifier(random_state=resample_id)
    elif name == "rocket_ds":
        return RocketClassifierDS(random_state=resample_id)
    elif name == "mini-rocket":
        return RocketClassifier(random_state=resample_id, rocket_transform="minirocket")
    elif name == "rocket_i":
        return RocketClassifier(random_state=resample_id, rocket_transform="rocket_i")
    elif name == "rocket_d":
        return RocketClassifier(random_state=resample_id, rocket_transform="rocket_d")
    elif name == "multi-rocket":
        return RocketClassifier(
            random_state=resample_id, rocket_transform="multirocket"
        )
    elif name == "arsenal":
        return Arsenal(random_state=resample_id, save_transformed_data=train_file)
    elif name == "mini-arsenal":
        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="minirocket",
        )
    elif name == "multi-arsenal":
        return Arsenal(
            random_state=resample_id,
            save_transformed_data=train_file,
            rocket_transform="multirocket",
        )
    # Shapelet based
    elif name == "stc" or name == "shapelettransformclassifier":
        return ShapeletTransformClassifier(
            transform_limit_in_minutes=120,
            random_state=resample_id,
            save_transformed_data=train_file,
        )
    else:
        raise Exception("UNKNOWN CLASSIFIER")
