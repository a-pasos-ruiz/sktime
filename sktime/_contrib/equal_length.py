from scipy import signal
import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts
from sktime.datasets import write_dataframe_to_tsfile as write_ts

def convert_to_equal_length(dataset, class_label, size):
    X_train, y_train = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TRAIN.ts")
    X_test, y_test = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TEST.ts")

    for key, value in X_train.iterrows():
        # print(key, value)
        for key2, value2 in value.iteritems():
            #  print(key, key2)
            X_train.loc[key][key2] = pd.Series(np.round(signal.resample(X_train.iloc[key][key2], size), 4))

    for key, value in X_test.iterrows():
        for key2, value2 in value.iteritems():
            X_test.loc[key][key2] = pd.Series(np.round(signal.resample(X_test.iloc[key][key2], size), 4))



    write_ts(
        X_train,
        "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
        dataset + "Eq",
        class_label,
        y_train,
        True,
        size,
        False, None, "_TRAIN")

    write_ts(
        X_test,
        "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
        dataset + "Eq",
        class_label,
        y_test,
        True,
        size,
        False, None, "_TEST")


def print_size(dataset):
    X_train, y_train = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TRAIN.ts")
    X_test, y_test = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TEST.ts")

    sizes = []

    for key, value in X_train.iterrows():
       # print(key, value)
        for key2, value2 in value.iteritems():
            print(key2, len(value2))
            sizes.append(len(value2))


    for key, value in X_test.iterrows():
        #print(key, value)
        for key2, value2 in value.iteritems():
            print(key2, len(value2))
            sizes.append(len(value2))

    return int(np.mean(sizes))

size = print_size("InsectWingbeat")
print(size)
convert_to_equal_length("InsectWingbeat",
                        ["Aedes_female", "Aedes_male", "Fruit_flies", "House_flies", "Quinx_female", "Quinx_male", "Stigma_female", "Stigma_male", "Tarsalis_female", "Tarsalis_male"],
                        size)
