# Train Output Preprocessing


import pandas as pd
import numpy as np

TrDataOut = np.zeros((4000,1),dtype='bool_')
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt'
data = pd.read_csv(path)
datanp = np.array(data)
for i in range(datanp.shape[0]):
    TrDataOut[i,0] = datanp[i,-1]
print(TrDataOut)
a_file = open("YTrain.txt", "w")
np.savetxt(a_file, TrDataOut)

# Test Output Preprocessing

TestDataOut = np.zeros((4000,1),dtype='bool_')
path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-b.txt'
data = pd.read_csv(path)
datanp = np.array(data)
for i in range(datanp.shape[0]):
    TestDataOut[i,0] = datanp[i,-1]
print(TestDataOut)
a_file = open("YTest.txt", "w")
np.savetxt(a_file, TestDataOut)
