from sklearn import svm

import pandas as pd
import numpy as np
import os

# Initialize the Train data
DataType = ['Temp', 'HR', 'NIDiasABP', 'NISysABP', 'NIMAP', 'Urine', 'SaO2', 'MechVent','RespRate','GCS','Albumin','ALP','ALT','AST','Bilirubin','BUN',\
            'Cholesterol','Creatinine','DiasABP','FiO2','Glucose','HCO3','HCT','Lactate','K','Mg','MAP','Na','PaCO2','PaO2','pH','Platelets','TropI',\
            'TropT','WBC']
count = len(DataType)
TrData = np.zeros((4000,count),dtype='float16')
FileIndex = -1

# Initialize train files
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-a/'
files = os.listdir(path)    # List all the files in the "path"
files.sort()

for file in files:
    data = pd.read_csv(path + file)
    datanp = np.array(data)
    lebal = DataType.copy()
    FileIndex = FileIndex + 1
    for i in range(datanp.shape[0]):
        if datanp[(datanp.shape[0] - i - 1), 1] in lebal:
            locate = lebal.index(datanp[(datanp.shape[0] - i - 1), 1])
            TrData[FileIndex, locate] = datanp[(datanp.shape[0] - i - 1), 2]
            lebal[locate] = '0'

# print(TrData)
# a_file = open("XTrain.txt", "w")
# np.savetxt(a_file, TrData)



# Test Data Preprocessing

# Initialize the Train data
count = len(DataType)
TestData = np.zeros((4000,count),dtype='float16')
FileIndex = -1

# Initialize train files
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-b/'
files = os.listdir(path)    # List all the files in the "path"
files.sort()

for file in files:
    data = pd.read_csv(path + file)
    datanp = np.array(data)
    LabelName = DataType.copy()
    FileIndex = FileIndex + 1
    for i in range(datanp.shape[0]):
        if datanp[(datanp.shape[0] - i - 1), 1] in LabelName:
            locate = LabelName.index(datanp[(datanp.shape[0] - i - 1), 1])
            TestData[FileIndex, locate] = datanp[(datanp.shape[0] - i - 1), 2]
            LabelName[locate] = '0'



TrDataOut = np.zeros((4000,1),dtype='bool_')
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt'
data = pd.read_csv(path)
datanp = np.array(data)
for i in range(datanp.shape[0]):
    TrDataOut[i,0] = datanp[i,-1]
# TrDataOut = TrDataOut.T
# print(TrDataOut.shape)
# print(TrData.shape)
# print(TestData)
# a_file = open("YTrain.txt", "w")
# np.savetxt(a_file, TrDataOut)

# Test Output Preprocessing

TestDataOut = np.zeros((4000,1),dtype='bool_')
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-b.txt'
data = pd.read_csv(path)
datanp = np.array(data)
for i in range(datanp.shape[0]):
    TestDataOut[i,0] = datanp[i,-1]
# print(TestDataOut.shape)

# print(TestDataOut)
# a_file = open("YTest.txt", "w")
# np.savetxt(a_file, TestDataOut)

# print(TestData)
# a_file = open("XTest.txt", "w")
# np.savetxt(a_file, TestData)

# XTraData = pd.read_csv('XTrain.txt')
# XTraData = np.array(XTraData, dtype='float16')
# # XTraData = float(XTraData)
#
# YTraData = pd.read_csv('YTrain.txt')
# YTraData = np.array(YTraData)
# # YTraData = float(YTraData)
# print(TestData[2,:])



# clf = svm.SVC(kernel='poly',degree=3,gamma='auto',probability=True)
# clf = svm.SVC(kernel='sigmoid',gamma='auto',probability=True) # 12 100 556 3332
# clf = svm.SVC(kernel='poly',degree=1,gamma='auto',probability=True)  # 12 5 555 3427
# clf = svm.SVC(kernel='poly',degree=3,gamma='auto',probability=True) # 219 480 349 2952
clf = svm.SVC(kernel='poly',degree=2,gamma='auto',probability=True)
clf.fit(TrData, TrDataOut.ravel())
a =0
b = 0
c = 0
d = 0
# print(TestData[85,:])
for i in range(4000):
    if clf.predict([TestData[i,:]])==True:
        if clf.predict([TestData[i,:]])==TestDataOut[i,0]:
            a = a + 1
        else:
            b = b + 1
    elif clf.predict([TestData[i,:]])==False:
        if clf.predict([TestData[i,:]])==TestDataOut[i,0]:
            d = d + 1
        else:
            c = c + 1
print('a=',a,'\t b=',b,'c=',c,'\t d=',d)
#         print(clf.predict([TestData[i,:]]))
#         print(i+2)

# for i in range(4000):
#     print(clf.predict([TestData[i,:]]))
#     print(i+2)




# print(clf)
# clf.predict([[ 37.6,92.,77.,148.,100.7]])
