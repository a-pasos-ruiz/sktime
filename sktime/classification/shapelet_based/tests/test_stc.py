# -*- coding: utf-8 -*-
"""ShapeletTransformClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
<<<<<<< HEAD
from sktime.datasets import load_japanese_vowels, load_unit_test, load_basic_motions
=======
from sktime.datasets import load_basic_motions, load_unit_test
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20


def test_stc_on_unit_test_data():
    """Test of ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train STC
<<<<<<< HEAD
    rotf = RotationForest(n_estimators=10)
    stc = ShapeletTransformClassifier(
        n_shapelets_considered=100,
        max_shapelets=10,
        random_state=0,
        estimator=rotf,
=======
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=3),
        max_shapelets=20,
        n_shapelet_samples=500,
        batch_size=100,
        random_state=0,
        save_transformed_data=True,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
    )
    stc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = stc.predict_proba(X_test.iloc[indices])
<<<<<<< HEAD
    testing.assert_array_equal(probas, stc_unit_test_probas)
=======
    testing.assert_array_almost_equal(probas, stc_unit_test_probas, decimal=2)
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20

    # test train estimate
    train_probas = stc._get_train_probs(X_train, y_train)
    train_preds = stc.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


<<<<<<< HEAD
def test_contracted_stc_on_unit_test_data():
    """Test of contracted ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    # train contracted STC
    stc = ShapeletTransformClassifier(
        time_limit_in_minutes=0.025,
        random_state=0,
    )
    stc.fit(X_train, y_train)

    assert accuracy_score(y_test, stc.predict(X_test)) >= 0.75


def test_stc_on_japanese_vowels():
=======
# def test_contracted_stc_on_unit_test_data():
#     """Test of contracted ShapeletTransformClassifier on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#
#     # train contracted STC
#     stc = ShapeletTransformClassifier(
#         estimator=RotationForest(contract_max_n_estimators=3),
#         max_shapelets=20,
#         time_limit_in_minutes=0.25,
#         contract_max_n_shapelet_samples=500,
#         batch_size=100,
#         random_state=0,
#     )
#     stc.fit(X_train, y_train)
#
#     assert accuracy_score(y_test, stc.predict(X_test)) >= 0.75


def test_stc_on_basic_motions():
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
    """Test of ShapeletTransformClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
<<<<<<< HEAD
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train STC
    rotf = RotationForest(n_estimators=10)
    stc = ShapeletTransformClassifier(
        n_shapelets_considered=100,
        max_shapelets=10,
        random_state=0,
        estimator=rotf,
=======
    indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)

    # train STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=3),
        max_shapelets=20,
        n_shapelet_samples=500,
        batch_size=100,
        random_state=0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
    )
    stc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
<<<<<<< HEAD
    probas = stc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stc_basic_motions_probas)
=======
    probas = stc.predict_proba(X_test.iloc[indices[:10]])
    testing.assert_array_almost_equal(probas, stc_basic_motions_probas, decimal=2)
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20


stc_unit_test_probas = np.array(
    [
        [
<<<<<<< HEAD
            0.3,
            0.7,
        ],
        [
            0.8,
            0.2,
=======
            0.0,
            1.0,
        ],
        [
            0.3333333333333333,
            0.6666666666666666,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
        ],
        [
            0.0,
            1.0,
        ],
        [
<<<<<<< HEAD
            0.9,
            0.1,
=======
            1.0,
            0.0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
<<<<<<< HEAD
            0.7,
            0.3,
=======
            1.0,
            0.0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
        ],
    ]
)
stc_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
<<<<<<< HEAD
            0.5,
            0.5,
            0.0,
            0.0,
=======
            0.6666666666666666,
            0.0,
            0.0,
            0.3333333333333333,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
        ],
        [
            0.0,
            0.0,
<<<<<<< HEAD
            0.6,
            0.4,
        ],
        [
            0.2,
            0.7,
            0.0,
            0.1,
=======
            1.0,
            0.0,
        ],
        [
            0.6666666666666666,
            0.3333333333333333,
            0.0,
            0.0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
        ],
        [
            0.0,
            0.0,
<<<<<<< HEAD
            0.2,
            0.8,
        ],
        [
            0.0,
            0.1,
            0.3,
            0.6,
        ],
        [
            0.7,
            0.3,
=======
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
            0.0,
            0.0,
        ],
        [
            0.0,
<<<<<<< HEAD
            0.1,
            0.6,
            0.3,
        ],
        [
            0.3,
            0.6,
            0.0,
            0.1,
        ],
        [
            0.3,
            0.7,
=======
            0.0,
            1.0,
            0.0,
        ],
        [
            0.3333333333333333,
            0.3333333333333333,
            0.0,
            0.3333333333333333,
        ],
        [
            0.0,
            1.0,
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
            0.0,
            0.0,
        ],
    ]
)


<<<<<<< HEAD
def print_array(array):
    print("[")
    for sub_array in array:
        print("[")
        for value in sub_array:
            print(value.astype(str), end="")
            print(", ")
        print("],")
    print("]")


if __name__ == "__main__":
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    rf = RandomForestClassifier(n_estimators=10)
    stc_u = ShapeletTransformClassifier(
        # n_shapelets=100,
        transform_limit_in_minutes=0.025,
        random_state=0,
        estimator=rf,
    )

    stc_u.fit(X_train, y_train)
    probas = stc_u.predict_proba(X_test.iloc[indices])
    print_array(probas)

    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    rf = RandomForestClassifier(n_estimators=10)
    stc_m = ShapeletTransformClassifier(
        n_shapelets=100,
        random_state=0,
        estimator=rf,
    )

    stc_m.fit(X_train.iloc[indices], y_train[indices])
    probas = stc_m.predict_proba(X_test.iloc[indices])
    print_array(probas)
=======
# def print_array(array):
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     stc_u = ShapeletTransformClassifier(
#         estimator=RotationForest(n_estimators=3),
#         max_shapelets=20,
#         n_shapelet_samples=500,
#         batch_size=100,
#         random_state=0,
#     )
#
#     stc_u.fit(X_train, y_train)
#     probas = stc_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)
#
#     stc_m = ShapeletTransformClassifier(
#         estimator=RotationForest(n_estimators=3),
#         max_shapelets=20,
#         n_shapelet_samples=500,
#         batch_size=100,
#         random_state=0,
#     )
#
#     stc_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = stc_m.predict_proba(X_test.iloc[indices[:10]])
#     print_array(probas)
>>>>>>> f351d25a21d4793568b3aafb85aa2089278ccd20
