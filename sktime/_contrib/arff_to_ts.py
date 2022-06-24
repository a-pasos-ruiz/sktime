
from sktime.datasets import load_from_arff_to_dataframe as load_ts
from sktime.datasets import write_dataframe_to_tsfile as write_ts


dataset = "Tiselac"
class_label = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
size = 23

X_train, y_train = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_arff\\" + dataset + "\\" + dataset + "_TRAIN.arff")
X_test, y_test = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_arff\\" + dataset + "\\" + dataset + "_TEST.arff")

write_ts(
    X_train,
    "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
    dataset ,
    class_label,
    y_train,
    True,
    size,
    False, None, "_TRAIN")

write_ts(
    X_test,
    "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
    dataset ,
    class_label,
    y_test,
    True,
    size,
    False, None, "_TEST")