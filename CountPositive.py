import numpy as np
import pandas as pd

path = './predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-b.txt'
data = pd.read_csv(path)
datanp = np.array(data)
count = 0
for i in range(datanp.shape[0]):
    if datanp[i,-1] == True:
        count = count + 1
print (count)