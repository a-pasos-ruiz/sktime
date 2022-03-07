import csv
import numpy as np

datasets = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "DuckDuckGeese",
    #	"EigenWorms",
    "EMO",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    #	"FaceDetection",
    "FingerMovements",
    "HAR",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    #    "InsectWingbeatEq",
    "JapaneseVowelsEq",
    "Libras",
    "LSST",
    "MotorImagery",
    "MindReading",
    "NATOPS",
    "PenDigitsEq",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "Siemens",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigitsEq",
    "StandWalkJump",
    "UWaveGestureLibrary"
]
resample = 30
path = "C:\\Users\\fbu19zru\\code\\results_final\\"
algorithm = "hc2-ds-ecp"


def read_file(dataset, resample):
    with open(path + algorithm + "\\Predictions\\" + dataset + "\\testResample" + str(resample) + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = [r for r in csv_reader]
        num_dimensions = rows[1][0][rows[1][0].find(':') + 1:]
        num_dimensions_selected = rows[1][1][rows[1][1].find(':') + 1:]
        train_time = rows[1][2][rows[1][2].find(':') + 1:]

        return [int(num_dimensions), int(num_dimensions_selected), int(train_time)]


def read_dataset(dataset):
    data = [read_file(dataset, i) for i in range(1, 30)]
    data2 = [x[1] / x[0] for x in data]
    data3 = [x[2] for x in data]
    print(dataset, np.mean(data2), np.mean(data3))



for dataset in datasets:
    read_dataset(dataset)
