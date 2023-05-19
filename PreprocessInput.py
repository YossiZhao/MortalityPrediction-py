# # Train Data Preprocessing

import pandas as pd
import numpy as np
import os

# Initialize the Train data
TrDataType = ['Temp', 'HR', 'NIDiasABP', 'NISysABP', 'NIMAP', 'Urine', 'SaO2', 'MechVent','RespRate','GCS','Albumin','ALP','ALT','AST','Bilirubin','BUN',\
            'Cholesterol','Creatinine','DiasABP','FiO2','Glucose','HCO3','HCT','Lactate','K','Mg','MAP','Na','PaCO2','PaO2','pH','Platelets','TropI',\
            'TropT','WBC']
count = len(TrDataType)
TrData = np.zeros((4000,count),dtype='float16')
FileIndex = -1

# Initialize train files
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-a/'
files = os.listdir(path)    # List all the files in the "path"
files.sort()

for file in files:
    data = pd.read_csv(path + file)
    datanp = np.array(data)
    lebal = TrDataType.copy()
    FileIndex = FileIndex + 1
    for i in range(datanp.shape[0]):
        if datanp[(datanp.shape[0] - i - 1), 1] in lebal:
            locate = lebal.index(datanp[(datanp.shape[0] - i - 1), 1])
            TrData[FileIndex, locate] = datanp[(datanp.shape[0] - i - 1), 2]
            lebal[locate] = '0'

print(TrData)
a_file = open("XTrain.txt", "w")
np.savetxt(a_file, TrData)



# Test Data Preprocessing

# Initialize the Train data
TestDataType = ['Temp', 'HR', 'NIDiasABP', 'NISysABP', 'NIMAP', 'Urine', 'SaO2', 'MechVent','RespRate','GCS','Albumin','ALP','ALT','AST','Bilirubin','BUN',\
            'Cholesterol','Creatinine','DiasABP','FiO2','Glucose','HCO3','HCT','Lactate','K','Mg','MAP','Na','PaCO2','PaO2','pH','Platelets','TropI',\
            'TropT','WBC']
count = len(TestDataType)
TestData = np.zeros((4000,count),dtype='float16')
FileIndex = -1

# Initialize train files
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-b/'
files = os.listdir(path)    # List all the files in the "path"
files.sort()

for file in files:
    data = pd.read_csv(path + file)
    datanp = np.array(data)
    LabelName = TestDataType.copy()
    FileIndex = FileIndex + 1
    for i in range(datanp.shape[0]):
        if datanp[(datanp.shape[0] - i - 1), 1] in LabelName:
            locate = LabelName.index(datanp[(datanp.shape[0] - i - 1), 1])
            TestData[FileIndex, locate] = datanp[(datanp.shape[0] - i - 1), 2]
            LabelName[locate] = '0'

print(TestData)
a_file = open("XTest.txt", "w")
np.savetxt(a_file, TestData)