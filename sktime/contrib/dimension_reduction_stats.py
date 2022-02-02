import csv

datasets = [
    "ArticularyWordRecognition",
   # "AtrialFibrillation",
   # "BasicMotions",
   # "Cricket",
    "DuckDuckGeese",
   # "EigenWorms",
  #  "EMO",
   # "Epilepsy",
   # "EthanolConcentration",
   # "ERing",
   # "FaceDetection",
    "FingerMovements",
   # "HAR",
    "HandMovementDirection",
   # "Handwriting",
    "Heartbeat",
   # "InsectWingbeat",
   # "JapaneseVowelsEq",
   # "Libras",
   # "LSST",
   # "MotorImagery",
   # "MindReading",
    "NATOPS",
   # "PenDigits",
   # "PEMS-SF",
   # "PhonemeSpectra",
   # "RacketSports",
    "Siemens",
   # "SelfRegulationSCP1",
   # "SelfRegulationSCP2",
  #  "SpokenArabicDigitsEq",
   # "StandWalkJump",
   # "UWaveGestureLibrary"
]

classifiers = ["hc2ds"]

resamples = 30

path = "C:\\Users\\fbu19zru\\code\\results_final\\"

for classifier in classifiers:
    for dataset in datasets:
        for resample in range(resamples):
            with open(path + classifier + "\\Predictions\\" + dataset + "\\testResample" + str(resample) + ".csv", 'r') as read_obj:
                reader = csv.reader(read_obj)
                line = next((x for i, x in enumerate(reader) if i == 1), None)
                print(classifier, dataset, resample, line[1], line[2])
